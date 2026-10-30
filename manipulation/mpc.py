"""Adapter to the unmodified DeepMind predictive-sampling planner on CPU."""

import hashlib
import importlib.util
import json
from pathlib import Path

import mujoco
import numpy as np

from .control import external_contacts, grasp_detected

ROOT = Path(__file__).resolve().parents[1]


def load():
    manifest = json.loads((ROOT / "manipulation/MPC_UPSTREAM.json").read_text())
    path = ROOT / "third_party/mujoco-mpc-python/predictive_sampling.py"
    if not path.exists():
        raise RuntimeError("Run python3 -m scripts.setup_mpc first")
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]["predictive_sampling.py"]:
        raise ValueError("Upstream sampler checksum mismatch")
    spec = importlib.util.spec_from_file_location("upstream_predictive_sampling", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PhysicsRefiner:
    def __init__(self, model, active, robot_dofs, reference, samples=12):
        self.active = active
        self.reference = reference
        self.nominal = np.zeros(model.nu)
        self.robot_dofs = robot_dofs
        self.updates = 0
        self.change = []
        joint = model.joint("case_free").id
        self.object_adr = model.jnt_qposadr[joint]

        # reward receives the upstream candidate's controls; bound them and freeze
        # unobserved joints before forwarding physics. No case force is applied.
        def physical_reward(m, d):
            candidate = d.ctrl[active].copy()
            d.ctrl[:] = self.nominal
            candidate = np.clip(candidate, self.nominal[active] - 0.06, self.nominal[active] + 0.06)
            candidate[:7] = np.clip(
                candidate[:7], self.nominal[active[:7]] - 0.003, self.nominal[active[:7]] + 0.003
            )
            d.ctrl[active] = np.clip(candidate, *m.actuator_ctrlrange[active].T)
            mujoco.mj_forward(m, d)
            d.qfrc_applied[:] = 0
            d.qfrc_applied[robot_dofs] = d.qfrc_bias[robot_dofs]
            difference = d.qpos[self.object_adr : self.object_adr + 3] - reference(d.time)
            contact_penalty = 0.0
            if 5 <= d.time < 15 and not grasp_detected(external_contacts(m, d)):
                contact_penalty = 0.5
            return (
                -2000 * float(difference @ difference)
                - 10 * float(np.sum((d.ctrl[active] - self.nominal[active]) ** 2))
                - 0.0005 * float(d.qvel @ d.qvel)
                - contact_penalty
            )

        upstream = load()
        self.planner = upstream.Planner(
            model,
            physical_reward,
            horizon=0.12,
            splinestep=0.04,
            planstep=0.002,
            nsample=samples,
            noise_scale=0.008,
            nimprove=1,
            interp="linear",
        )

    def improve(self, data, nominal):
        self.nominal = nominal.copy()
        # Seed every horizon with the current contact-feedback command. Sampling
        # refines this seed through actual forward dynamics, rather than IK alone.
        self.planner.policy._parameters[:] = nominal[:, None]
        self.planner.policy._times[:] = data.time + np.arange(self.planner.policy._nspline) * 0.04
        self.planner.improve_policy(
            data.qpos, data.qvel, data.act, data.time, data.mocap_pos, data.mocap_quat
        )
        selected = self.planner.action_from_policy(data.time)[self.active]
        selected = np.clip(selected, nominal[self.active] - 0.06, nominal[self.active] + 0.06)
        selected[:7] = np.clip(
            selected[:7], nominal[self.active[:7]] - 0.003, nominal[self.active[:7]] + 0.003
        )
        selected = np.clip(selected, *self.planner._model.actuator_ctrlrange[self.active].T)
        self.updates += 1
        self.change.append(float(np.linalg.norm(selected - nominal[self.active])))
        return selected
