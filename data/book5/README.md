# Book5 recording and inputs

Recorded by the project author on an iPhone 14, handheld in the left hand.
The right hand gestures, grasps a black case, carries and turns it, places it
on the table, releases it and makes final gestures. The clip has 748 frames
at approximately 59.96 Hz (12.5 seconds).

The author supplied dimensions: square book 22 × 22 × 2 cm, case 6 × 6 × 2 cm,
and right hand approximately 23 cm from wrist crease to middle fingertip.
Mass and friction are simulation assumptions, specified in
[the configuration](../../configs/book5.json).

[Video](video.mp4) is the full recording with audio and container metadata
removed; the video stream was copied without re-encoding.
[Provenance](provenance.json) records original and published checksums.
The author authorized publication of this recording and generated previews.

## Inputs

| Path under `inputs/` | Contents and use |
|---|---|
| `hand-track.npz` | Raw FastHaMeR/MANO estimates; 683 valid detections out of 748 |
| `hand-registration/` | Image/support/contact-refined hand proposal, selected frames and residual report |
| `hand-prior.npz` | Registered landmarks and six fitted finger commands, used for gestures/turn anchors |
| `initial-gestures.npz` | Reviewed initial 181-frame arm/hand/waist fit |
| `camera-observations.json` | Book-corner observations for metric camera calibration |
| `masks/000000.png` | Initial case segmentation for scene alignment |
| `grasp-prior.npz` | RH56-derived arm/palm path and thumb/index pre-shape seeds |
| `grasp-sources.json` | Upstream revisions and attribution for that prior |
| `object-track.npz` | Timestamped rigid-pose candidates, validity, component IDs and review flags |
| `tracking-anchor/` | Initial reconstructed case pose and sampled surface |
| `tracking-map/` | Textured reconstruction coordinate map and geometry |
| `tracking-provenance.json` | Source lineage for reconstructed inputs |

Raw object candidates contain 740 valid rows: 430 connected to the initial
anchor, 310 flagged for review and eight gaps. Coverage is not pose accuracy.
Missing observations remain missing; command interpolation is separately labelled.

The tracking map uses its own 198 × 129 × 18 mm book coordinate scale. Physical
scene generation uses the supplied 220 × 220 × 20 mm dimensions and an explicit
map-to-task adaptation. Hand-contact registration is inferred from images and
support assumptions; it does not measure tactile force.

[Input preparation](../../docs/preparation.md) explains how the observations
are obtained. [Methods](../../docs/methods.md) distinguishes them from calibrated
motion proposals and simulated measurements.
