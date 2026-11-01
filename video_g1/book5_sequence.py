"""Generate the reviewed Book5 motion proposal using native G1 arm and waist IK.

The phase boundaries below are video annotations, not automatically detected events.
Observed hand rotations and finger proposals drive the motion; the grasp comes
from the verified RH56 plan. Support corrections remain separate control proposals.
"""

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from humanoid.book_g1 import SCENE_ROTATION, make_model
from humanoid.kinematics import control_vector, set_configuration
from video_g1.geometry import palm_frame
from video_g1.ik import fit_upper_body
from video_g1.provenance import write_manifest
from video_g1.timing import retime_controls


def build(scene, hand_reference, grasp_reference, object_track, out):
    """Retain every source sample while fitting task phases in a feasible workspace.

    The demonstration uses initial gestures through frame 180, approach through
    255, closure through 310, settling through 330, transport through 390, turn
    through 570, placement through 600 and release/retraction thereafter.
    Translation amplitude and clearance are embodiment adaptations. Missing raw
    observations are never filled or rewritten by this function.
    """
    out.mkdir(parents=True, exist_ok=False)
    write_manifest(
        out,
        [
            hand_reference,
            grasp_reference,
            object_track,
            scene / "scene.xml",
            Path("data/book5/inputs/initial-gestures.npz"),
        ],
    )
    m, active, state = make_model(scene, out, scene)
    d = mujoco.MjData(m)
    r = np.load(hand_reference)
    h = r["target_landmarks_m"]
    g = np.load(grasp_reference)
    if [m.actuator(int(i)).name for i in active] != g["action_names"].tolist():
        raise ValueError("Grasp actuator order does not match the native model")
    closed = g["controls"][np.argmin(abs(g["time_s"] - 7))].copy()
    set_configuration(m, d, control_vector(m, active, closed))
    p = d.xpos[m.body("R_hand_base_link").id].copy()
    R = d.xmat[m.body("R_hand_base_link").id].reshape(3, 3).copy()
    c = d.xpos[m.body("case").id].copy()
    points = d.site_xpos[[m.site(f"landmark_{i}").id for i in range(21)]]
    native = palm_frame(points)
    base = palm_frame(h[315])
    align = native @ base.T
    # Demonstration-derived rotation proposals: reviewed phase anchors retain gaps separately.
    keys = np.array([330, 360, 390, 405, 420, 435, 450, 465, 480, 495, 510, 525, 540, 555, 570, 585, 600])
    rotations = []
    for i in keys:
        dr = align @ palm_frame(h[i if i > 330 else 315]) @ base.T @ align.T
        if i >= 585:
            dr = Rotation.from_euler("z", -0.5).as_matrix()
        rotations.append(dr)
    rotations = np.array(rotations)
    frames = np.arange(748)
    drs = Slerp(keys, Rotation.from_matrix(rotations))(np.clip(frames, 330, 600)).as_matrix()
    # Goal position derives from the source object track; support and clearance are control proposals.
    track = np.load(object_track)
    delta = SCENE_ROTATION @ (
        np.nanmedian(track["translation_world_m"][-30:], axis=0) - track["translation_world_m"][0]
    )
    delta[:2] *= 22 / 19.8
    delta[2] = -0.02
    positions = np.array([c + delta * np.clip((i - 330) / 60, 0, 1) for i in frames])
    positions[:, 2] = c[2] + np.interp(
        frames, [0, 330, 370, 550, 585, 600, 747], [0, 0, 0.10, 0.10, 0.04, -0.02, -0.02]
    )
    actions = np.tile(closed, (748, 1))
    extras = np.zeros((748, 3))
    errors = []
    # Keep initial gesture finger shapes. Source-global hand translation is shifted into the robot workspace.
    N = native.T @ R
    gestures_path = Path("data/book5/inputs/initial-gestures.npz")
    gestures = np.load(gestures_path)
    if gestures["action_names"].tolist() != [m.actuator(int(i)).name for i in active]:
        raise ValueError("Initial gesture actuator mapping differs")
    actions[:181] = gestures["actions"]
    extras[:181] = gestures["extra_actions"]
    # Move above the scene before rotating into the grasp. A direct joint-space
    # blend from the palm-up gesture can sweep the case off the book.
    ctrl = control_vector(m, active, actions[180])
    waist_ids = np.array([m.actuator(f"waist_{a}_joint").id for a in ["yaw", "roll", "pitch"]])
    ctrl[waist_ids] = extras[180]
    set_configuration(m, d, ctrl)
    start_p = d.xpos[m.body("R_hand_base_link").id].copy()
    start_r = d.xmat[m.body("R_hand_base_link").id].reshape(3, 3).copy()
    set_configuration(m, d, control_vector(m, active, g["controls"][0]))
    end_p = d.xpos[m.body("R_hand_base_link").id].copy()
    end_r = d.xmat[m.body("R_hand_base_link").id].reshape(3, 3).copy()
    transition_r = Slerp([190, 205], Rotation.from_matrix([start_r, end_r]))
    v = np.r_[actions[180, :7], extras[180]]
    # Native verified approach/closure, followed by two execution seconds of force settling.
    for i in range(181, 331):
        if i <= 210:
            if i <= 190:
                pos = start_p + np.array([0, 0, 0.28]) * (i - 180) / 10
                rot = start_r
            elif i <= 205:
                f = (i - 190) / 15
                pos = (1 - f) * (start_p + [0, 0, 0.28]) + f * (end_p + [0, 0, 0.08])
                rot = transition_r(i).as_matrix()
            else:
                f = (i - 205) / 5
                pos = end_p + np.array([0, 0, 0.08]) * (1 - f)
                rot = end_r
            fingers = (1 - np.clip((i - 180) / 25, 0, 1)) * actions[180, 7:] + np.clip(
                (i - 180) / 25, 0, 1
            ) * g["controls"][0, 7:]
            v, pe, re = fit_upper_body(m, active, v, fingers, pos, rot)
            actions[i] = np.r_[v[:7], fingers]
            extras[i] = v[7:]
            errors.append([i, pe, re])
        else:
            if i <= 225:
                f = (i - 210) / 15
                actions[i] = (1 - f) * actions[210] + f * g["controls"][0]
                extras[i] = (1 - f) * extras[210]
            else:
                t = np.interp(i, [225, 255, 310, 330], [0, 2, 5, 7])
                actions[i] = [np.interp(t, g["time_s"], g["controls"][:, j]) for j in range(13)]
    v = np.r_[closed[:7], 0.0, 0.0, 0.0]
    for i in range(333, 601, 3):
        dr = drs[i]
        pos = positions[i] + dr @ (p - c)
        v, pe, re = fit_upper_body(m, active, v, closed[7:], pos, dr @ R)
        actions[i] = np.r_[v[:7], closed[7:]]
        extras[i] = v[7:]
        errors.append([i, pe, re])
    for j in range(13):
        actions[330:601, j] = np.interp(np.arange(330, 601), np.arange(330, 601, 3), actions[330:601:3, j])
    for j in range(3):
        extras[330:601, j] = np.interp(np.arange(330, 601), np.arange(330, 601, 3), extras[330:601:3, j])
    # Release above the table, retract, then retain final observed wrist/finger gestures.
    final_align = drs[600] @ native @ palm_frame(h[630]).T
    last_final_observation = 670
    for i in range(601, 748):
        actions[i] = actions[600]
        extras[i] = extras[600]
        f = np.clip((i - 610) / 25, 0, 1)
        for finger in range(int(g["grasp_fingers"])):
            motor = [11, 10, 9, 8, 7][finger]
            actions[i, motor] = (1 - f) * actions[600, motor]
        if i >= 635:
            dr = drs[600]
            pos = (
                positions[600] + dr @ (p - c) + np.array([-0.04, -0.02, 0.12]) * np.clip((i - 635) / 35, 0, 1)
            )
            fingers = actions[i, 7:].copy()
            rot = dr @ R
            if i >= 670:
                fingers = r["finger_commands_rad"][i].copy()
                if np.isfinite(h[i]).all():
                    last_final_observation = i
                observed_r = final_align @ palm_frame(h[last_final_observation]) @ N
                blend = np.clip((i - 670) / 10, 0, 1)
                rot = (
                    rot
                    @ Rotation.from_rotvec(
                        blend * Rotation.from_matrix(rot.T @ observed_r).as_rotvec()
                    ).as_matrix()
                )
                pos += 0.5 * (h[last_final_observation, 0] - h[630, 0])
            v, pe, re = fit_upper_body(m, active, v, fingers, pos, rot)
            actions[i] = np.r_[v[:7], fingers]
            extras[i] = v[7:]
            errors.append([i, pe, re])
    # Minimum phase durations reproduce the verified closure; all motion is additionally slew retimed.
    base_times = np.interp(
        frames,
        [0, 180, 210, 240, 310, 330, 390, 510, 600, 610, 635, 670, 747],
        [0, 5, 7, 9, 12, 14, 20, 30, 37, 38, 40, 43, 47],
    )
    times = retime_controls(base_times, np.c_[actions, extras], 0.8)
    slow_gestures = retime_controls(base_times[:181], np.c_[actions[:181], extras[:181]], 0.45)
    dt = np.diff(times)
    dt[:180] = np.maximum(dt[:180], np.diff(slow_gestures))
    times = np.r_[0, np.cumsum(dt)]
    waist = np.array([m.actuator(f"waist_{a}_joint").id for a in ["yaw", "roll", "pitch"]])
    planned_landmarks = []
    for action, extra in zip(actions, extras):
        ctrl = control_vector(m, active, action)
        ctrl[waist] = extra
        set_configuration(m, d, ctrl)
        planned_landmarks.append(d.site_xpos[[m.site(f"landmark_{i}").id for i in range(21)]].copy())
    phase = np.select(
        [
            frames <= 180,
            frames <= 255,
            frames <= 310,
            frames <= 330,
            frames <= 390,
            frames <= 570,
            frames <= 610,
            frames <= 635,
        ],
        ["gesture", "approach", "grasp", "confirm", "transport", "turn", "place", "release"],
        default="retract-and-gesture",
    )
    np.savez_compressed(
        out / "reference.npz",
        **{
            k: r[k]
            for k in r.files
            if k
            in [
                "source_frame",
                "source_time_s",
                "observed_hand",
                "target_landmarks_m",
                "raw_camera_registered_landmarks_m",
            ]
        },
        actions=actions,
        execution_time_s=times,
        active_actuators=active,
        state_joints=state,
        extra_actuators=waist,
        phase=phase,
        grasp_force_target_N=float(g["grasp_force_target_N"]) if "grasp_force_target_N" in g else 0.8,
        grasp_finger_ids=np.arange(int(g["grasp_fingers"])),
        confirmation_finger_ids=np.arange(int(g["grasp_fingers"])),
        control_phase_frames=[310, 330, 350, 600, 610],
        control_phase_names=["force-start", "carry-start", "confirmation-deadline", "carry-end", "force-end"],
        planned_landmarks_m=planned_landmarks,
        action_names=[m.actuator(int(i)).name for i in active],
        extra_action_names=[m.actuator(int(i)).name for i in waist],
        extra_actions=extras,
        target_case_position_m=positions,
        target_case_rotation=drs @ Rotation.from_matrix(d.xmat[m.body("case").id].reshape(3, 3)).as_matrix(),
        object_observation_valid=track["valid"],
        object_observation_requires_review=track["requires_review"],
    )
    (out / "fit.json").write_text(
        json.dumps(
            {
                "errors_source_frame_position_m_rotation_rad": errors,
                "duration_s": float(times[-1]),
                "maximum_position_error_m": float(np.max(np.array(errors)[:, 1])),
                "maximum_rotation_error_deg": float(np.max(np.array(errors)[:, 2]) * 180 / np.pi),
            },
            indent=2,
        )
    )


def main():
    """Generate Book5 controls from saved observations without neural inference."""
    import argparse

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--hands", type=Path, default=Path("data/book5/inputs/hand-prior.npz"))
    parser.add_argument("--grasp", type=Path, required=True)
    parser.add_argument("--objects", type=Path, default=Path("data/book5/inputs/object-track.npz"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.scene, args.hands, args.grasp, args.objects, args.output)


if __name__ == "__main__":
    main()
