# Optional RGB object tracking

The default reconstruction uses geometric and hand-relative priors because a
small dark case is heavily covered by fingers. Optional native experiments test
whether image evidence can replace parts of that prior.

- **ViSP moving edges:** fast CPU updates, but reliable tracking is lost during
  pickup/carry in this clip. Rejected poses are never promoted into robot motion.
- **M3T RGB regions:** follows some visible motion after an initial mesh/pose
  seed. Estimated depth and orientation drift under occlusion. Seven conservative
  image-centre corrections survive the hybrid gates; prior depth/orientation and
  fallback frames remain explicit.

Mask overlap is evaluated with reference-informed image regions, so it is not
independent pose ground truth. Accepted outlines do not prove correct depth or
rotation. Attempts and fusion evidence are retained in the current reconstruction
folders; `reconstruction/BOOK_POSE_UPSTREAM.json` pins source revisions and licenses.

The adapters are `native/book_pose_tracker` and `native/book_m3t_tracker`.
Build outputs belong under ignored `native/build` and are not distributed.
Building requires downloading the exact pinned upstream sources at the manifest's
source paths, plus the OpenCV/Eigen/GLEW/GLFW/OpenMP dependencies described by the
CMake files. The tested M3T build generates a forward-compatible OpenGL 4.1 copy
for macOS and omits the unused texture modality for OpenCV 5 compatibility.

Run an installed tracker or reconstruct the stored hybrid:

```bash
OMP_NUM_THREADS=3 .venv-book-reconstruction/bin/python -m reconstruction.book_visp \
  runs/new-m3t --backend m3t
.venv-book-reconstruction/bin/python -m reconstruction.book_visp \
  runs/new-visp --shape box
.venv-book-reconstruction/bin/python -m reconstruction.book_pose_fusion \
  runs/new-fusion --attempts runs/book5-m3t-forward-01 runs/book5-m3t-backward-01 \
  --render
```

The simulator does not need either tracker to replay/audit the retained physical
reference. Fully automatic unknown-object six-dimensional recovery remains
unresolved in this single-view occluded sequence.
