"""Search collision-aware antipodal grasps and verify them with a free case.

Human landmarks provide a posture prior, not observed fingertip-contact labels.
Each candidate is checked by actuator-driven closure and lift, with external
finger-force feedback. Case identity is used only for evaluation.
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares, minimize

from humanoid.book_g1 import make_model
from humanoid.kinematics import set_configuration
from manipulation.control import finger_geoms
from manipulation.fit import control_vector

from .grasp_verify import verify


def fit_candidate(model, active, seed, human, lateral=0.0):
    """Fit side opposition while keeping native collision geometry clear of supports."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    case = model.geom("case_solid").id
    centre = data.geom_xpos[case].copy()
    rotation = data.geom_xmat[case].reshape(3, 3)
    half = model.geom_size[case]
    normal = rotation[:, 1]
    tangent = rotation[:, 0]
    targets = np.array(
        [
            centre - normal * half[1],
            centre + normal * half[1] + tangent * 0.014,
            centre + normal * half[1] - tangent * 0.01,
        ]
    )
    targets[:, 2] += 0.003
    targets += tangent * lateral
    groups = finger_geoms(model)
    supports = [model.geom(n).id for n in ["table_collision", "book_collision"]]
    hand = [
        g
        for g in range(model.ngeom)
        if model.geom_contype[g] and model.body(model.geom_bodyid[g]).name.startswith("R_")
    ]
    sites = [model.site(f"landmark_{i}").id for i in range(21)]
    low, high = model.actuator_ctrlrange[active].T

    def residual(u):
        set_configuration(model, data, control_vector(model, active, u))
        mujoco.mj_forward(model, data)
        distances = []
        for geoms in groups[:3]:
            found = []
            for geom in geoms:
                witness = np.zeros(6)
                distance = mujoco.mj_geomDistance(model, data, geom, case, 0.15, witness)
                found.append(distance)
            distances.append(min(found))
        support = np.array(
            [mujoco.mj_geomDistance(model, data, g, s, 0.1, None) for g in hand for s in supports]
        )
        return np.r_[
            0.15 * (data.site_xpos[sites] - human).ravel(),
            3 * (data.site_xpos[np.array(sites)[[4, 8, 12]]] - targets).ravel(),
            8 * (np.array(distances) - 0.0005),
            20 * np.minimum(distances, 0),
            20 * np.minimum(support - 0.001, 0),
            0.001 * (u - seed),
        ]

    fit = least_squares(
        residual,
        np.clip(seed, low + 1e-6, high - 1e-6),
        bounds=(low + 1e-7, high - 1e-7),
        max_nfev=160,
        jac="3-point",
        diff_step=1e-3,
    )
    polished = minimize(
        lambda q: float(np.dot(residual(q), residual(q))),
        np.clip(fit.x, low + 1e-6, high - 1e-6),
        method="Powell",
        bounds=list(zip(low + 1e-6, high - 1e-6)),
        options={"maxfev": 3500, "xtol": 1e-4, "ftol": 1e-6},
    )
    q = polished.x if polished.fun < np.dot(residual(fit.x), residual(fit.x)) else fit.x
    error = residual(q)
    return q, float(np.dot(error, error))


def main():
    """Test a bounded set of geometric proposals and keep all physical outcomes."""
    p = argparse.ArgumentParser(__doc__)
    for name in ["scene", "reference", "seed", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--posture-frame", type=int, default=240)
    p.add_argument("--candidates", type=int, default=12)
    p.add_argument("--verify", type=int, default=4)
    a = p.parse_args()
    if a.candidates < 1 or a.verify < 1:
        p.error("Positive candidate and verification counts required")
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    model, active, state = make_model(a.scene, a.output, a.scene)
    human = np.load(a.reference / "reference.npz")["target_landmarks_m"][a.posture_frame]
    if not np.isfinite(human).all():
        raise ValueError("Posture prior requires a valid observed/refined hand frame")
    raw = np.load(a.seed)
    seed = raw["initial_actions"].copy()
    seed[7:11] = [0.7, 0.6, 0.4, 0.35]
    seed[11:] = [0.23, 0.45]
    rng = np.random.default_rng(1)
    proposals = []
    for i in range(a.candidates):
        initial = seed.copy()
        if i:
            initial += rng.normal(0, np.r_[np.full(7, 0.12), np.full(6, 0.15)])
        q, cost = fit_candidate(model, active, initial, human, lateral=[0, 0.006, -0.006][i % 3])
        proposals.append((cost, q))
        print(f"Geometry {i + 1}/{a.candidates}: {cost:.6f}", flush=True)
    reports = []
    best = None
    for index, (cost, q) in enumerate(sorted(proposals, key=lambda item: item[0])[: a.verify]):
        report, rollout = verify(model, active, state, q)
        report.update(candidate=index, geometric_cost=cost, actions=q.tolist())
        reports.append(report)
        folder = a.output / f"candidate-{index:02d}"
        folder.mkdir()
        np.savez_compressed(folder / "rollout.npz", **rollout)
        (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        ranking = (report["success"], report["normal_opposition_supported_lift_s"], -cost)
        if best is None or ranking > best[0]:
            best = (ranking, q, index, report["success"])
    np.savez_compressed(a.output / "grasp.npz", actions=best[1], active=active, state=state)
    report = {
        "method": "Antipodal side-contact search with native support-clearance constraints; source posture prior",
        "source_frame_prior": a.posture_frame,
        "prior_is_not_contact_label": True,
        "tested": reports,
        "selected_candidate": best[2],
        "success": bool(best[3]),
        "elapsed_s": time.monotonic() - started,
        "ready_for_training": False,
    }
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
