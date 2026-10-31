"""Keep every source sample while explicitly retiming control proposals."""

import numpy as np


def fill_control_gaps(times, observed_controls, valid):
    """Interpolate commands only; return provenance codes without changing observations.

    Codes: 0 observed-frame fit, 1 interior command interpolation, 2 boundary hold.
    These commands are not additional human pose observations.
    """
    times = np.asarray(times, dtype=float)
    observed_controls = np.asarray(observed_controls, dtype=float)
    valid = np.asarray(valid, bool)
    if (
        times.ndim != 1
        or observed_controls.ndim != 2
        or observed_controls.shape[0] != len(times)
        or valid.shape != times.shape
    ):
        raise ValueError("Timeline, control and validity shapes differ")
    if not np.isfinite(times).all() or not (np.diff(times) > 0).all():
        raise ValueError("Finite, strictly increasing timestamps required")
    if not np.isfinite(observed_controls[valid]).all():
        raise ValueError("Observed controls must be finite")
    if not valid.any():
        raise ValueError("At least one observed fit is required")
    ids = np.flatnonzero(valid)
    result = np.column_stack(
        [np.interp(times, times[ids], observed_controls[ids, j]) for j in range(observed_controls.shape[1])]
    )
    kind = np.zeros(len(times), np.uint8)
    kind[~valid] = 1
    kind[np.arange(len(times)) < ids[0]] = 2
    kind[np.arange(len(times)) > ids[-1]] = 2
    return result, kind


def retime_controls(source_time, controls, max_speed):
    """Return increasing execution times; never drop, reorder or speed up source samples."""
    source_time = np.asarray(source_time, dtype=float)
    controls = np.asarray(controls, dtype=float)
    if (
        source_time.ndim != 1
        or controls.ndim != 2
        or len(source_time) != len(controls)
        or len(source_time) < 2
    ):
        raise ValueError("At least two aligned time/control samples required")
    if not np.isfinite(source_time).all() or not np.isfinite(controls).all():
        raise ValueError("Time and control samples must be finite")
    speed = np.broadcast_to(np.asarray(max_speed, dtype=float), controls.shape[1:])
    if not np.isfinite(speed).all() or (speed <= 0).any() or not (np.diff(source_time) > 0).all():
        raise ValueError("Positive speed and strictly increasing source time required")
    needed = np.max(np.abs(np.diff(controls, axis=0)) / speed, axis=1)
    dt = np.maximum(np.diff(source_time), needed)
    return np.r_[0, np.cumsum(dt)]
