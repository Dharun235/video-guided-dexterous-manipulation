"""Fit every recorded hand pose jointly, with explicit temporal regularization.

This generates a kinematic control reference. It does not demonstrate a physical
manipulation and never replaces missing human observations with fitted labels.
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix, diags, kron, vstack
from scipy.spatial.transform import Rotation

from humanoid.book_g1 import make_model
from humanoid.kinematics import set_configuration
from manipulation.fit import control_vector

from .geometry import coupled_control_map, palm_frame, site_jacobians
from .observations import world_hands
from .provenance import write_manifest
from .timing import fill_control_gaps, retime_controls


class TrajectoryObjective:
    """Sparse landmark/palm trajectory residual and analytic coupled-hand Jacobian."""

    def __init__(
        self, model, active, targets, observed, times, velocity_weight=0.02, acceleration_weight=0.0005
    ):
        self.model = model
        self.active = active
        self.data = mujoco.MjData(model)
        self.targets = np.nan_to_num(targets)
        self.observed = observed
        self.n = len(times)
        self.a = len(active)
        self.sites = [model.site(f"landmark_{i}").id for i in range(21)]
        self.mapping = coupled_control_map(model, active)
        self.palm_targets = np.array(
            [palm_frame(x) if ok else np.eye(3) for x, ok in zip(self.targets, observed)]
        )
        self.weights = np.ones(21)
        self.weights[0] = 4
        self.weights[[4, 8, 12, 16, 20]] = 1.5
        self.local_size = 63 + 60 + 9
        dt = float(np.median(np.diff(times)))
        identity = diags(np.ones(self.a))
        first = diags([-np.ones(self.n - 1), np.ones(self.n - 1)], [0, 1], shape=(self.n - 1, self.n))
        second = diags(
            [np.ones(self.n - 2), -2 * np.ones(self.n - 2), np.ones(self.n - 2)],
            [0, 1, 2],
            shape=(self.n - 2, self.n),
        )
        self.temporal = vstack(
            [
                kron(first, identity) * velocity_weight / np.sqrt(dt),
                kron(second, identity) * acceleration_weight / dt**1.5,
            ],
            format="csr",
        )
        self.rows = np.repeat(np.arange(self.n * self.local_size), self.a)
        self.columns = (
            np.arange(self.n)[:, None, None] * self.a
            + np.arange(self.a)[None, None, :]
            + np.zeros((1, self.local_size, 1), int)
        ).ravel()
        self.cached = None
        self.calls = 0

    def evaluate(self, flat):
        """Cache pose residuals and derivatives shared by the nonlinear solver callbacks."""
        if self.cached is not None and np.array_equal(flat, self.cached):
            return
        local = np.zeros((self.n, self.local_size))
        jac = np.zeros((self.n, self.local_size, self.a))
        self.points = np.empty((self.n, 21, 3))
        for i, u in enumerate(flat.reshape(self.n, self.a)):
            set_configuration(self.model, self.data, control_vector(self.model, self.active, u))
            points = self.data.site_xpos[self.sites].copy()
            self.points[i] = points
            if not self.observed[i]:
                continue
            point_jac = site_jacobians(self.model, self.data, self.sites, self.mapping)
            basis, basis_jac = palm_frame(points, point_jac)
            target = self.targets[i]
            local[i] = np.r_[
                ((points - target) * self.weights[:, None]).ravel(),
                0.35 * ((points[1:] - points[0]) - (target[1:] - target[0])).ravel(),
                0.04 * (basis - self.palm_targets[i]).ravel(),
            ]
            jac[i] = np.vstack(
                [
                    (point_jac * self.weights[:, None, None]).reshape(63, self.a),
                    0.35 * (point_jac[1:] - point_jac[0]).reshape(60, self.a),
                    0.04 * basis_jac.reshape(9, self.a),
                ]
            )
        self.local = local.ravel()
        self.jacobian = coo_matrix(
            (jac.ravel(), (self.rows, self.columns)), shape=(self.n * self.local_size, self.n * self.a)
        ).tocsr()
        self.jacobian.eliminate_zeros()
        self.cached = flat.copy()
        self.calls += 1

    def residual(self, flat):
        """Combine metre-space pose errors and time-scaled joint regularization."""
        self.evaluate(flat)
        return np.r_[self.local, self.temporal @ flat]

    def jac(self, flat):
        """Return the sparse analytic derivative used by SciPy's trust-region solver."""
        self.evaluate(flat)
        return vstack([self.jacobian, self.temporal], format="csr")


def run(a):
    """Write all source samples, observed-pose fits, command provenance and timing map."""
    started = time.monotonic()
    a.output.mkdir(parents=True, exist_ok=False)
    write_manifest(
        a.output,
        [a.hands / "hand_track.npz", a.cameras / "cameras.npz", a.scene / "scene.xml"]
        + ([a.refined_hands / "observations.npz"] if a.refined_hands else []),
    )
    hands = np.load(a.hands / "hand_track.npz")
    cameras = np.load(a.cameras / "cameras.npz")
    hand_scale = 1.0
    scale_report = {"method": "Unmeasured MANO scale", "scale": 1.0}
    if a.hand_length_m is not None:
        local = hands["joints_hand_centered_m"]
        straight = np.linalg.norm(local[:, 12] - local[:, 0], axis=1)
        chain = np.linalg.norm(np.diff(local[:, [0, 9, 10, 11, 12]], axis=1), axis=2).sum(axis=1)
        flat = hands["valid"] & (straight / chain > 0.97)
        if not flat.any() or a.hand_length_m <= 0:
            raise ValueError("Positive hand length and observed extended-hand frames required")
        hand_scale = a.hand_length_m / np.median(straight[flat])
        scale_report = {
            "method": "Approximate user wrist-crease to middle-fingertip length / median extended model length",
            "measured_hand_length_m": a.hand_length_m,
            "extended_observations": int(flat.sum()),
            "model_extended_length_m": float(np.median(straight[flat])),
            "scale": float(hand_scale),
            "limitation": "Model wrist and anatomical crease may differ; no accuracy guarantee",
        }
    target, observed, _reprojection = world_hands(hands, cameras, hand_scale)
    m, active, state = make_model(a.scene, a.output, a.calibration)
    raw_target = target.copy()
    landmark_source = "Camera-registered human estimates"
    if a.refined_hands is not None:
        refined = np.load(a.refined_hands / "observations.npz")
        if not np.array_equal(refined["source_frame"], hands["source_frame"]):
            raise ValueError("Refined hand timeline differs from source")
        target = refined["landmarks_m"].copy()
        observed = refined["valid"].copy()
        landmark_source = "Refined registration with assumed table and inferred contacts; requires review"

    registration = {"method": "Camera registration only", "translation_m": [0, 0, 0]}
    if a.anchor_frame is not None:
        frame = a.anchor_frame
        if not observed[frame]:
            raise ValueError("Task anchor must have an observed hand")
        d0 = mujoco.MjData(m)
        mujoco.mj_forward(m, d0)
        centre = d0.xpos[m.body("case").id].copy()
        shift = centre - target[frame, [4, 8]].mean(axis=0)
        target = target + shift
        registration = {
            "method": "Constant task-space translation from reviewed grasp frame; not a corrected human observation",
            "anchor_frame": frame,
            "translation_m": shift.tolist(),
            "contact_prior": "Thumb/index midpoint aligned to case centre",
        }
    times = hands["seconds"]
    lo, hi = m.actuator_ctrlrange[active].T
    if a.seed:
        old = np.load(a.seed / "retarget.npz")
        seed, _ = fill_control_gaps(times, old["actions"], old["valid"])
    else:
        seed = np.tile(
            np.clip([-0.5, -0.2, 0, 1, 0, 0, 0, 0.3, 0.3, 0.3, 0.3, 0.2, 0.5], lo + 1e-6, hi - 1e-6),
            (len(times), 1),
        )
    # Re-fit each observed pose to the corrected camera gauge before global fitting.
    work = mujoco.MjData(m)
    sites = [m.site(f"landmark_{i}").id for i in range(21)]
    mapping = coupled_control_map(m, active)
    previous = None
    for i in np.flatnonzero(observed):
        anchor = seed[i].copy() if previous is None else previous.copy()

        def error(u, i=i, anchor=anchor):
            set_configuration(m, work, control_vector(m, active, u))
            return np.r_[(work.site_xpos[sites] - target[i]).ravel(), 0.01 * (u - anchor)]

        def derivative(u):
            set_configuration(m, work, control_vector(m, active, u))
            return np.vstack(
                [site_jacobians(m, work, sites, mapping).reshape(63, len(active)), 0.01 * np.eye(len(active))]
            )

        fit = least_squares(
            error,
            np.clip(anchor, lo + 1e-6, hi - 1e-6),
            jac=derivative,
            bounds=(lo + 1e-7, hi - 1e-7),
            max_nfev=25,
        )
        seed[i] = fit.x
        previous = fit.x.copy()
    seed, _ = fill_control_gaps(times, seed, observed)
    objective = TrajectoryObjective(m, active, target, observed, times)
    fit = least_squares(
        objective.residual,
        np.clip(seed, lo + 1e-6, hi - 1e-6).ravel(),
        jac=objective.jac,
        bounds=(np.tile(lo + 1e-7, len(times)), np.tile(hi - 1e-7, len(times))),
        max_nfev=a.iterations,
        ftol=1e-5,
        xtol=1e-5,
        gtol=1e-5,
        x_scale="jac",
        verbose=2,
    )
    actions, origin = fill_control_gaps(times, fit.x.reshape(len(times), len(active)), observed)
    objective.evaluate(actions.ravel())
    fitted = objective.points.copy()
    error = np.linalg.norm(fitted - target, axis=2)
    palm_error = np.full(len(times), np.nan)
    for i in np.flatnonzero(observed):
        palm_error[i] = Rotation.from_matrix(palm_frame(fitted[i]).T @ palm_frame(target[i])).magnitude()
    execution = retime_controls(times, actions, np.r_[np.full(4, 1.5), np.full(3, 3.0), np.full(6, 2.0)])
    np.savez_compressed(
        a.output / "reference.npz",
        source_frame=hands["source_frame"],
        source_time_s=times,
        execution_time_s=execution,
        observed_hand=observed,
        control_origin=origin,
        actions=actions,
        target_landmarks_m=target,
        raw_camera_registered_landmarks_m=raw_target,
        fitted_landmarks_m=fitted,
        landmark_error_m=error,
        palm_error_rad=palm_error,
        active_actuators=active,
        state_joints=state,
    )
    report = {
        "source_frames": len(times),
        "observed_frames": int(observed.sum()),
        "all_source_frames_preserved": True,
        "source_duration_s": float(times[-1]),
        "execution_duration_s": float(execution[-1]),
        "mean_wrist_error_m": float(np.nanmean(error[:, 0])),
        "mean_tip_error_m": float(np.nanmean(error[:, [4, 8, 12, 16, 20]])),
        "median_palm_error_deg": float(np.nanmedian(palm_error) * 180 / np.pi),
        "p95_palm_error_deg": float(np.nanpercentile(palm_error, 95) * 180 / np.pi),
        "largest_control_step_rad": float(np.max(abs(np.diff(actions, axis=0)))),
        "observation_filling": False,
        "control_origin_codes": {
            "0": "observed-pose fit",
            "1": "interpolated command; no observation",
            "2": "boundary command hold; no observation",
        },
        "landmark_source": landmark_source,
        "raw_hand_observed_frames": int(hands["valid"].sum()),
        "hand_scale": scale_report,
        "task_registration": registration,
        "physical_execution": False,
        "ready_for_training": False,
        "calibration_report": str(a.cameras / "report.json"),
        "elapsed_s": time.monotonic() - started,
        "solver_evaluations": fit.nfev,
        "solver_success": bool(fit.success),
    }
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (a.output / "joint-map.json").write_text(
        json.dumps(
            {
                "action_names": [m.actuator(int(i)).name for i in active],
                "state_joint_names": [m.joint(int(i)).name for i in state],
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(report, indent=2))


def main():
    """Fit the complete observed timeline, without prescribed manipulation phases."""
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--hands", type=Path, required=True)
    p.add_argument("--cameras", type=Path, required=True)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument(
        "--anchor-frame", type=int, help="Reviewed grasp frame for explicit task-space registration"
    )
    p.add_argument(
        "--hand-length-m", type=float, help="Approximate measured wrist-to-middle-tip length in metres"
    )
    p.add_argument(
        "--refined-hands", type=Path, help="Explicit diagnostic contact/support-refined observation proposal"
    )
    p.add_argument("--seed", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=30)
    run(p.parse_args())


if __name__ == "__main__":
    main()
