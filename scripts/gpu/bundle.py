"""Package a portable G1/Inspire model and honest reference data for GPU retargeting."""

import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from humanoid.book_g1 import BACKGROUND_CONTROL


def main():
    """Package portable model assets, reference controls and explicit physics assumptions."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--rollout", type=Path, default=Path("runs/transfer-final/rollout"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path((args.rollout / "scene-path.txt").read_text().strip())
    data = np.load(args.rollout / "rollout.npz")
    tree = ET.parse(source)
    root = tree.getroot()
    assets = args.output / "assets"
    assets.mkdir()
    copied = {}
    for node in root.findall("asset/*"):
        filename = node.get("file")
        if filename:
            path = Path(filename)
            if not path.is_absolute():
                path = source.parent / path
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            name = digest[:12] + "_" + path.name
            if name not in copied:
                shutil.copy2(path, assets / name)
                copied[name] = digest
            node.set("file", "assets/" + name)
    # Encode robot-only gravity compensation in MJCF so CPU and GPU use the same
    # physical model. The source controller also cancelled Coriolis; this adapter
    # does not silently assume that the two are dynamically equivalent.
    for body in root.find(".//body[@name='pelvis']").iter("body"):
        body.set("gravcomp", "1")
    root.find("option").set("timestep", ".002")
    tree.write(args.output / "scene.xml")
    m = mujoco.MjModel.from_xml_path(str((args.output / "scene.xml").resolve()))
    schema = json.loads((args.rollout / "schema.json").read_text())
    active = np.array([m.actuator(n).id for n in schema["action_names"]])
    qpos = data["qpos"]
    times = data["time_s"]
    qvel = np.zeros((len(qpos), m.nv))
    for i in range(1, len(qpos)):
        mujoco.mj_differentiatePos(m, qvel[i], times[i] - times[i - 1], qpos[i - 1], qpos[i])
    qvel[0] = qvel[1]
    ctrl = np.zeros((len(times), m.nu))
    ctrl[:, active] = data["actions"]
    for name, value in BACKGROUND_CONTROL.items():
        ctrl[:, m.actuator(name).id] = value
    np.savez_compressed(
        args.output / "reference.npz",
        qpos=qpos,
        qvel=qvel,
        ctrl=ctrl,
        time_s=times,
        active_actuators=active,
        phase=data["phase"],
    )
    meta = dict(
        reference="Retimed task-adaptation baseline, not full video tracking",
        source_rollout_sha256=hashlib.sha256((args.rollout / "rollout.npz").read_bytes()).hexdigest(),
        nq=m.nq,
        nv=m.nv,
        nu=m.nu,
        ref_dt=float(np.median(np.diff(times))),
        physics_change="Robot-only MJCF gravcomp replaces source qfrc_bias compensation; CPU/GPU equivalence must be tested",
        derivatives="Full reference qvel estimated with mj_differentiatePos between saved 50 Hz states",
        assets_sha256=copied,
        source_report=json.loads((args.rollout / "report.json").read_text()),
        object_body_id=m.body("case").id,
        object_qpos_adr=int(m.jnt_qposadr[m.joint("case_free").id]),
        no_object_actuators=True,
        no_object_welds=True,
        requires_review=True,
    )
    (args.output / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({key: meta[key] for key in ["nq", "nv", "nu", "ref_dt"]}, indent=2))


if __name__ == "__main__":
    main()
