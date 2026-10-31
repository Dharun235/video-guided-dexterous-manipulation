"""Run SPIDER's unmodified sampling optimizer and Warp backend on a fixed-root G1.

The CPU MuJoCo plant receives only robot motor commands. Its object is never
reset after initialization. GPU worlds are candidate rollouts, synchronized to
that plant, not reference-pose playback.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np


def main():
    """Refine motor commands on GPU and execute them on an independent CPU plant."""
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--spider", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=1)
    p.add_argument("--start", type=float, default=0)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--iterations", type=int, default=2)
    p.add_argument("--replay-only", action="store_true")
    p.add_argument("--horizon", type=float, default=0.16)
    p.add_argument("--slew", type=float, default=1.2)
    a = p.parse_args()
    import torch
    import warp as wp

    a.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(a.spider.resolve()))
    from spider.config import Config, compute_steps
    from spider.io import load_data
    from spider.optimizers.sampling import make_optimize_fn, make_optimize_once_fn, make_rollout_fn
    from spider.simulators import mjwp as backend

    started = time.monotonic()
    torch.manual_seed(0)
    np.random.seed(0)
    manifest = json.loads((a.input / "manifest.json").read_text())
    raw = np.load(a.input / "reference.npz")
    active = raw["active_actuators"]
    cfg = Config(
        embodiment_type="right",
        robot_type="g1_inspire",
        device="cuda:0",
        sim_dt=0.002,
        ctrl_dt=0.08,
        ref_dt=0.02,
        horizon=a.horizon,
        knot_dt=a.horizon / 4,
        num_samples=a.samples,
        max_num_iterations=a.iterations,
        use_torch_compile=False,
        contact_guidance=False,
        num_dyn=1,
        num_dr=1,
        terminate_resample=False,
        contact_rew_scale=0.0,
        show_viewer=False,
        save_video=False,
        save_config=False,
    )
    cfg.model_path = str((a.input / "scene.xml").resolve())
    cfg.data_path = str((a.input / "reference.npz").resolve())
    cfg.nq = manifest["nq"]
    cfg.nv = manifest["nv"]
    cfg.nu = manifest["nu"]
    cfg.nq_obj = 7
    cfg.npair = 0
    cfg.base_pos_rew_scale = 0.02
    cfg.base_rot_rew_scale = 0.02
    cfg.joint_rew_scale = 0.02
    cfg.pos_rew_scale = 1.0
    cfg.rot_rew_scale = 0.05
    cfg.vel_rew_scale = 0.0001
    cfg.improvement_threshold = 0.0
    compute_steps(cfg)
    cfg.beta_traj = 0.3 ** (1 / max(a.iterations, 1))
    cfg.env_params_list = [[{}] for _ in range(a.iterations)]
    cfg.noise_scale = torch.zeros(
        (a.samples, int(round(cfg.horizon / cfg.knot_dt)), cfg.nu), device=cfg.device
    )
    cfg.noise_scale[:, :, active[:7]] = 0.012
    cfg.noise_scale[:, :, active[7:]] = 0.04
    cfg.noise_scale[0] = 0.0
    ref = load_data(cfg, cfg.data_path)
    print("Reference and upstream optimizer ready", flush=True)
    # right/left backend indexing is the all-hinge robot + final free-object layout;
    # humanoid_object assumes a floating pelvis, which this supported G1 lacks.
    if not a.replay_only:
        env = backend.setup_env(cfg, ref)
        print("GPU worlds compiled", flush=True)

        def bounded_step(config, environment, ctrl):
            low = torch.as_tensor(
                environment.model_cpu.actuator_ctrlrange[:, 0], device=config.device, dtype=torch.float32
            )
            high = torch.as_tensor(
                environment.model_cpu.actuator_ctrlrange[:, 1], device=config.device, dtype=torch.float32
            )
            bounded = torch.clamp(ctrl, min=low, max=high)
            previous = wp.to_torch(environment.data_wp.ctrl)
            bounded = torch.maximum(
                torch.minimum(bounded, previous + a.slew * config.sim_dt), previous - a.slew * config.sim_dt
            )
            return backend.step_env(config, environment, bounded)

        rollout = make_rollout_fn(
            bounded_step,
            backend.save_state,
            backend.load_state,
            backend.get_reward,
            backend.get_terminal_reward,
            backend.get_terminate,
            backend.get_trace,
            backend.save_env_params,
            backend.load_env_params,
            backend.copy_sample_state,
        )
        optimize = make_optimize_fn(make_optimize_once_fn(rollout))
        m = env.model_cpu
    else:
        m = backend.setup_mj_model(cfg)
    d = mujoco.MjData(m)
    first = int(round(a.start / 0.02))
    d.qpos[:] = raw["qpos"][first]
    d.qvel[:] = raw["qvel"][first]
    d.ctrl[:] = raw["ctrl"][first]
    d.time = a.start
    mujoco.mj_forward(m, d)
    object_adr = manifest["object_qpos_adr"]
    initial = d.qpos[object_adr : object_adr + 3].copy()

    def sync():
        for name, value in [
            ("qpos", d.qpos),
            ("qvel", d.qvel),
            ("qacc_warmstart", d.qacc_warmstart),
            ("time", np.array([d.time])),
            ("ctrl", d.ctrl),
        ]:
            dest = wp.to_torch(getattr(env.data_wp, name))
            src = torch.as_tensor(value, dtype=dest.dtype, device=cfg.device)
            dest.copy_(src.expand_as(dest))

    rows = []
    full_controls = []
    plan = None
    infos = []
    iteration = 0
    end = a.start + a.seconds
    while d.time < end - 1e-8:
        index = min(int(round(d.time / cfg.sim_dt)), len(ref[0]) - cfg.horizon_steps - 1)
        slice_ref = tuple(x[index : index + cfg.horizon_steps] for x in ref)
        if a.replay_only:
            command = slice_ref[2][0].cpu().numpy()
        else:
            sync()
            if plan is None:
                plan = slice_ref[2].clone()
            else:
                previous = plan[cfg.ctrl_steps :]
                new_ctrl = ref[2][index + len(previous) : index + len(previous) + cfg.ctrl_steps]
                plan = torch.cat([previous, new_ctrl], dim=0)
            command_tensor, info = optimize(cfg, env, plan, slice_ref)
            plan = command_tensor.detach().clone()
            # Upstream candidate zero is the current warm-start plan. Execute the
            # optimized control block on the independent CPU plant, then replan.
            command_tensor = command_tensor.cpu().numpy()
            infos.append(
                {
                    k: np.asarray(v).tolist()
                    for k, v in info.items()
                    if k not in ["trace_sample", "trace_cost"]
                }
            )
        count = min(cfg.ctrl_steps, int(round((end - d.time) / cfg.sim_dt)))
        for j in range(count):
            if a.replay_only:
                idx = min(int(round(d.time / 0.02)), len(raw["ctrl"]) - 1)
                command = raw["ctrl"][idx]
            else:
                command = command_tensor[j]
            bounded = np.clip(command, *m.actuator_ctrlrange.T)
            d.ctrl[:] += np.clip(bounded - d.ctrl, -a.slew * cfg.sim_dt, a.slew * cfg.sim_dt)
            mujoco.mj_step(m, d)
            full_controls.append(d.ctrl.copy())
            if not np.isfinite(d.qpos).all():
                raise RuntimeError("Nonfinite CPU plant")
            if len(rows) == 0 or d.time - rows[-1][0] >= 0.019 - 1e-8:
                rows.append((d.time, d.qpos.copy(), d.qvel.copy(), d.ctrl.copy()))
        iteration += 1
        if iteration % 10 == 0:
            print(
                f"Plant {d.time:.2f}s | elapsed {time.monotonic() - started:.1f}s | case z={d.qpos[object_adr + 2]:.3f}",
                flush=True,
            )
    np.savez_compressed(
        a.output / "trajectory.npz",
        time_s=[r[0] for r in rows],
        qpos=[r[1] for r in rows],
        qvel=[r[2] for r in rows],
        ctrl=[r[3] for r in rows],
        executed_ctrl_2ms=full_controls,
    )
    actual = np.array([r[1][object_adr : object_adr + 3] for r in rows])
    idx = np.clip(np.round(np.array([r[0] for r in rows]) / 0.02).astype(int), 0, len(raw["qpos"]) - 1)
    error = np.linalg.norm(actual - raw["qpos"][idx, object_adr : object_adr + 3], axis=1)
    report = dict(
        reference=manifest["reference"],
        optimizer="Unmodified SPIDER sampling optimizer + MuJoCo Warp candidate backend",
        warm_start="Shift previous optimized horizon and append reference tail, as in upstream runner",
        horizon_s=a.horizon,
        full_command_stream_recorded=True,
        replay_only=a.replay_only,
        gpu_worlds=0 if a.replay_only else a.samples,
        iterations=a.iterations,
        start_s=a.start,
        duration_s=a.seconds,
        elapsed_s=time.monotonic() - started,
        mean_reference_object_error_m=float(error.mean()),
        maximum_reference_object_error_m=float(error.max()),
        maximum_lift_m=float(actual[:, 2].max() - initial[2]),
        object_pose_set_only_at_reset=True,
        object_external_forces=False,
        object_attachments=False,
        robot_gravity_compensation="MJCF robot-only gravcomp",
        control_target_slew_limit_rad_s=a.slew,
        ready_for_training=False,
        requires_cpu_replay=True,
        requires_review=True,
    )
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (a.output / "optimization.json").write_text(json.dumps(infos) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
