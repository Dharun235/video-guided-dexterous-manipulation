# Full-video hand extraction

![Estimated right hand over the source recording](preview.gif)

All 748 frames are processed. A right hand is detected in 683 frames (91.3%).
The 65 missing detections remain explicitly invalid; no hidden poses are filled.
See [coverage](coverage.json) for frame intervals. Detection coverage measures
availability, not pose accuracy. The mesh overlay is an estimate, especially
where fingers or the object hide one another.

Generated with `reconstruction.video_hands` from the published Book5 recording
using FastHaMeR 0.8.0/Core ML and the user's licensed MANO model. The GIF samples
the full overlay video at 8 fps. MANO files and raw mesh arrays are not included.

The direct G1/Inspire IK baseline fits all 683 detected frames. Mean wrist and
fingertip residuals are 11.7 mm and 10.6 mm against inferred targets. See
[fit metrics](retarget-metrics.json). This is kinematic fitting without physical
contact verification; the earlier reference grasp result does not validate it.

[Source versus fitted robot video](kinematic-fit.mp4): kinematic playback only;
the object is static and missing detections show the previous pose with a warning.
