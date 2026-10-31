"""Package full-timeline motor proposals and uncertain object candidates for SPIDER.

Object candidates are re-registered through their original projected geometry.
This is a coordinate/geometry conversion, not a new pose measurement. Review
flags and missing poses remain in the source-indexed archive.
"""

import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from humanoid.book_g1 import ORIGIN, SCENE_ROTATION
from humanoid.kinematics import set_configuration
from manipulation.fit import control_vector


def convert_objects(objects, old_cameras, cameras, old_intrinsics, size):
    """Convert only valid pose candidates, retaining independent source uncertainty."""
    surface = np.load(objects / "case_surface.npy")
    track = np.load(objects / "object_track.npz")
    # Bounds corners preserve the candidate's projected rigid axes and extent.
    low, high = surface.min(0), surface.max(0)
    old = np.array(
        [[x, y, z] for x in [low[0], high[0]] for y in [low[1], high[1]] for z in [low[2], high[2]]]
    )
    new = (old - (low + high) / 2) / (high - low) * size
    count = len(track["valid"])
    positions = np.full((count, 3), np.nan)
    rotations = np.full((count, 3, 3), np.nan)
    residuals = np.full(count, np.nan)
    valid = track["valid"].copy()
    for i in np.flatnonzero(valid):
        world = old @ track["rotation_world"][i].T + track["translation_world_m"][i]
        old_r = np.array(old_cameras[i]["rotation"])
        old_t = np.array(old_cameras[i]["translation"])
        pixels = cv2.projectPoints(world, old_r, old_t, old_intrinsics, None)[0][:, 0]
        ok, r, t = cv2.solvePnP(new, pixels, cameras["intrinsics"], None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok or t[2, 0] <= 0:
            valid[i] = False
            continue
        r, t = cv2.solvePnPRefineLM(new, pixels, cameras["intrinsics"], None, r, t)
        cw = cameras["rotation_world_to_camera"][i]
        ct = cameras["translation_world_to_camera_m"][i]
        positions[i] = cw.T @ (t.ravel() - ct)
        rotations[i] = cw.T @ cv2.Rodrigues(r)[0]
        uv = cv2.projectPoints(new, r, t, cameras["intrinsics"], None)[0][:, 0]
        residuals[i] = np.mean(np.linalg.norm(uv - pixels, axis=1))
    return track, valid, positions, rotations, residuals


def run(a):
    """Create portable assets and a uniformly sampled control reference, not a rollout."""
    a.output.mkdir(parents=True, exist_ok=False)
    assets = a.output / "assets"
    assets.mkdir()
    tree = ET.parse(a.reference / "scene.xml")
    root = tree.getroot()
    hashes = {}
    for node in root.findall("asset/*"):
        if node.get("file"):
            path = Path(node.get("file"))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            name = digest[:12] + "_" + path.name
            if name not in hashes:
                shutil.copy2(path, assets / name)
                hashes[name] = digest
            node.set("file", "assets/" + name)
    for body in root.find(".//body[@name='pelvis']").iter("body"):
        body.set("gravcomp", "1")
    root.find("option").set("timestep", ".002")
    tree.write(a.output / "scene.xml")
    m = mujoco.MjModel.from_xml_path(str((a.output / "scene.xml").resolve()))
    d = mujoco.MjData(m)
    ref = np.load(a.reference / "reference.npz")
    c = np.load(a.cameras / "cameras.npz")
    cfg = json.loads(a.config.read_text())
    old_meta = json.loads(a.old_calibration.read_text())
    old_cameras = json.loads(a.old_cameras.read_text())
    track, valid, p, r, error = convert_objects(
        a.objects, old_cameras, c, np.array(old_meta["camera"]["intrinsics"]), np.array(cfg["case_size_m"])
    )
    initial = d.qpos[m.jnt_qposadr[m.joint("case_free").id] :][:3]
    support_offset = (initial - ORIGIN) @ SCENE_ROTATION - p[np.flatnonzero(valid)[0]]
    p = p + support_offset
    source = ref["source_time_s"]
    execution = ref["execution_time_s"]
    times = np.arange(0, execution[-1] + 0.04, 0.02)
    src = np.interp(times, execution, source)
    ids = np.flatnonzero(valid)
    pi = np.column_stack([np.interp(src, source[ids], p[ids, j]) for j in range(3)])
    ri = Slerp(source[ids], Rotation.from_matrix(r[ids]))(
        np.clip(src, source[ids[0]], source[ids[-1]])
    ).as_matrix()
    actions = np.column_stack([np.interp(times, execution, ref["actions"][:, j]) for j in range(13)])
    qpos = []
    ctrl = []
    adr = m.jnt_qposadr[m.joint("case_free").id]
    for i, u in enumerate(actions):
        command = control_vector(m, ref["active_actuators"], u)
        set_configuration(m, d, command)
        d.qpos[adr : adr + 3] = SCENE_ROTATION @ pi[i] + ORIGIN
        d.qpos[adr + 3 : adr + 7] = Rotation.from_matrix(SCENE_ROTATION @ ri[i]).as_quat()[[3, 0, 1, 2]]
        qpos.append(d.qpos.copy())
        ctrl.append(command)
    qpos = np.array(qpos)
    qvel = np.zeros((len(times), m.nv))
    for i in range(1, len(times)):
        mujoco.mj_differentiatePos(m, qvel[i], 0.02, qpos[i - 1], qpos[i])
    np.savez_compressed(
        a.output / "reference.npz",
        qpos=qpos,
        qvel=qvel,
        ctrl=ctrl,
        time_s=times,
        active_actuators=ref["active_actuators"],
        phase=np.zeros(len(times), int),
    )
    np.savez_compressed(
        a.output / "object-candidates.npz",
        source_frame=track["source_frame"],
        valid=valid,
        requires_review=track["requires_review"],
        component=track["component"],
        translation_world_m=p,
        rotation_world=r,
        conversion_residual_px=error,
    )
    manifest = {
        "reference": "Full-video hand reference plus converted uncertain object candidates; not ground truth",
        "nq": m.nq,
        "nv": m.nv,
        "nu": m.nu,
        "object_qpos_adr": int(adr),
        "object_body_id": m.body("case").id,
        "assets_sha256": hashes,
        "source_frames": len(source),
        "source_duration_s": float(source[-1]),
        "execution_duration_s": float(execution[-1]),
        "support_translation_correction_m": support_offset.tolist(),
        "object_missing_poses": int((~valid).sum()),
        "object_flagged_poses": int(track["requires_review"].sum()),
        "interpolation": "Control proposals only; source candidate validity and review flags preserved separately",
        "requires_review": True,
        "ready_for_training": False,
    }
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def main():
    """Package a full, explicitly uncertain video reference for bounded GPU experiments."""
    p = argparse.ArgumentParser(__doc__)
    for name in ["reference", "cameras", "config", "objects", "old-cameras", "old-calibration", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
