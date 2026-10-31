"""Run bounded position checks and a force-feedback negative control."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from video_g1.provenance import write_manifest
from video_g1.sequence import score, simulate


def audit(reference_path, output):
    """Change only the reset position or feedback switch; keep the same motor plan."""
    output.mkdir(parents=True, exist_ok=False)
    reference = dict(np.load(reference_path / "reference.npz"))
    reports = []
    for name, offset, feedback in [
        ("x-plus-3mm", 0.003, True),
        ("x-minus-3mm", -0.003, True),
        ("no-feedback", 0, False),
    ]:
        model = mujoco.MjModel.from_xml_path(str((reference_path / "scene.xml").resolve()))
        adr = model.jnt_qposadr[model.joint("case_free").id]
        # This changes the initial condition only; no case state is set during execution.
        model.qpos0[adr] += offset
        arrays = simulate(model, reference["active_actuators"], reference, output / name, feedback=feedback)
        report = score(reference, arrays)
        report.update(condition=name, case_reset_x_offset_m=offset, force_feedback_enabled=feedback)
        (output / name / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        reports.append(report)
    summary = {"conditions": reports, "statistical_success_rate": False, "nominal_plan_reused": True}
    (output / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_manifest(output, [reference_path / "reference.npz", reference_path / "scene.xml"])
    return summary


def main():
    """Save physical evidence for each controlled condition without modifying the reference."""
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(audit(args.reference, args.output), indent=2))


if __name__ == "__main__":
    main()
