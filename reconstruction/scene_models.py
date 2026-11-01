"""CPU perception adapters with explicit model files and no robot-task annotations."""

import cv2
import numpy as np


class Hands:
    """Detect hand landmarks on independent frames, including redetection after occlusion."""

    def __init__(self, folder):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self.mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(folder / "hand_landmarker.task")),
            num_hands=2,
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def __call__(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.detector.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb))
        return [
            dict(
                landmarks=[[p.x, p.y, p.z] for p in hand],
                handedness=result.handedness[i][0].category_name,
                handedness_score=result.handedness[i][0].score,
            )
            for i, hand in enumerate(result.hand_landmarks)
        ]

    def close(self):
        """Release the native detector resources."""
        self.detector.close()


class Segmenter:
    """Run pinned split SAM ONNX models with automatic point proposals on CPU."""

    def __init__(self, folder, variant="sam2"):
        import onnxruntime as ort

        self.variant = variant
        prefix = "sam2.1_hiera_tiny." if variant == "sam2" else "efficientsam_ti_"
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.encoder = ort.InferenceSession(
            str(folder / (prefix + "encoder.onnx")), options, providers=["CPUExecutionProvider"]
        )
        self.decoder = ort.InferenceSession(
            str(folder / (prefix + "decoder.onnx")), options, providers=["CPUExecutionProvider"]
        )

    def proposals(self, image, prompts):
        """Return deduplicated masks; predicted IoU is a ranking score, not a probability."""
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        if self.variant == "sam2":
            tensor = (cv2.resize(rgb, (1024, 1024)) - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        else:
            tensor = rgb
        tensor = tensor.transpose(2, 0, 1)[None].astype(np.float32)
        features = self.encoder.run(None, {self.encoder.get_inputs()[0].name: tensor})
        candidates = []
        for point in prompts:
            if self.variant == "sam2":
                feed = dict(zip([o.name for o in self.encoder.get_outputs()], features))
                feed.update(
                    point_coords=np.array([[point]], np.float32) / [w, h] * 1024,
                    point_labels=np.ones((1, 1), np.float32),
                    mask_input=np.zeros((1, 1, 256, 256), np.float32),
                    has_mask_input=np.zeros(1, np.float32),
                )
                feed["point_coords"] = feed["point_coords"].astype(np.float32)
                masks, scores = self.decoder.run(None, feed)[:2]
                masks, scores = masks[0], scores[0]
            else:
                feed = dict(
                    image_embeddings=features[0],
                    batched_point_coords=np.array([[[point]]], np.float32),
                    batched_point_labels=np.ones((1, 1, 1), np.float32),
                    orig_im_size=np.array([h, w], np.int64),
                )
                masks, scores = self.decoder.run(None, feed)[:2]
                masks, scores = masks[0, 0], scores[0, 0]
            for mask, score in zip(masks, scores):
                mask = cv2.resize(mask.astype(np.float32), (w, h)) > 0
                fraction = mask.mean()
                if score < 0.7 or not 0.001 < fraction < 0.4:
                    continue
                if any(
                    np.logical_and(mask, c["mask"]).sum() / max(1, np.logical_or(mask, c["mask"]).sum())
                    > 0.85
                    for c in candidates
                ):
                    continue
                candidates.append(dict(mask=mask, score=float(score)))
        return candidates


class MetricGeometry:
    """Infer MoGe-2 metric point maps on CPU; outputs remain learned estimates."""

    def __init__(self, folder):
        import torch
        from moge.model.v2 import MoGeModel

        self.torch = torch
        torch.set_num_threads(4)
        self.model = MoGeModel.from_pretrained(str(folder / "moge2-small.pt")).eval().to("cpu")

    def __call__(self, image):
        tensor = (
            self.torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()).permute(2, 0, 1).float()
            / 255
        )
        with self.torch.inference_mode():
            result = self.model.infer(tensor, resolution_level=0, use_fp16=False)
        return {k: v.cpu().numpy() for k, v in result.items()}


class People:
    """Exclude confidently segmented skin/clothing when ranking object masks."""

    def __init__(self, folder):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self.mp = mp
        self.detector = vision.ImageSegmenter.create_from_options(
            vision.ImageSegmenterOptions(
                base_options=python.BaseOptions(model_asset_path=str(folder / "selfie_multiclass.tflite")),
                output_category_mask=True,
                output_confidence_masks=True,
            )
        )

    def __call__(self, image):
        result = self.detector.segment(
            self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        )
        # Accessories are excluded from this rejection mask: they may be the task object.
        probability = sum(result.confidence_masks[i].numpy_view() for i in [1, 2, 3, 4])
        return np.squeeze(probability) > 0.8

    def close(self):
        """Release the native segmentation model."""
        self.detector.close()
