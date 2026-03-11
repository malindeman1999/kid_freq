from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from analysis_models import VNAScan


def _window_width_in_freq_units(freq: np.ndarray, width_ghz: float) -> float:
    abs_median = float(np.nanmedian(np.abs(freq)))
    if abs_median > 1e6:
        return width_ghz * 1e9  # frequency is likely in Hz
    if abs_median > 1e3:
        return width_ghz * 1e3  # frequency is likely in MHz
    return width_ghz  # frequency is likely in GHz


def _median_percentile_filter(
    freq: np.ndarray,
    amp: np.ndarray,
    width_ghz: float,
    step_ghz: float,
    retain_pct: float,
    center_pct: float,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    width = _window_width_in_freq_units(freq, width_ghz)
    step = _window_width_in_freq_units(freq, step_ghz)
    if width <= 0:
        raise ValueError("Window width must be > 0")
    if step <= 0:
        raise ValueError("Compute step must be > 0")

    retain_pct = float(np.clip(retain_pct, 0.1, 100.0))
    center_pct = float(np.clip(center_pct, 0.0, 100.0))
    low_pct = max(0.0, center_pct - 0.5 * retain_pct)
    high_pct = min(100.0, center_pct + 0.5 * retain_pct)

    n = len(freq)
    if n == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=float)

    # Compute median windows on a regular frequency grid (step) with window width.
    sort_idx = np.argsort(freq)
    inv_idx = np.empty_like(sort_idx)
    inv_idx[sort_idx] = np.arange(n)
    f_sorted = freq[sort_idx]
    a_sorted = amp[sort_idx]

    keep_sorted = np.zeros(n, dtype=bool)
    baseline_sorted = np.zeros(n, dtype=float)
    total_span = float(f_sorted[-1] - f_sorted[0]) if n > 1 else 0.0
    n_sections = max(1, int(np.floor(total_span / step)) + 1)

    half_width = 0.5 * width
    half_step = 0.5 * step
    for k in range(n_sections):
        center = float(f_sorted[0]) + k * step
        w_left = np.searchsorted(f_sorted, center - half_width, side="left")
        w_right = np.searchsorted(f_sorted, center + half_width, side="right")
        if w_right <= w_left:
            if progress_cb is not None:
                progress_cb(k + 1, n_sections)
            continue

        vals = a_sorted[w_left:w_right]
        med = float(np.median(vals))
        low_val = float(np.percentile(vals, low_pct))
        high_val = float(np.percentile(vals, high_pct))

        a_left = np.searchsorted(f_sorted, center - half_step, side="left")
        a_right = np.searchsorted(f_sorted, center + half_step, side="right")
        if a_right > a_left:
            assign_vals = a_sorted[a_left:a_right]
            baseline_sorted[a_left:a_right] = med
            keep_sorted[a_left:a_right] = (assign_vals >= low_val) & (assign_vals <= high_val)

        if progress_cb is not None:
            progress_cb(k + 1, n_sections)

    # Fill any unassigned points using nearest-neighbor baseline/keep fallback.
    unassigned = baseline_sorted == 0
    if np.any(unassigned):
        baseline_sorted[unassigned] = np.median(a_sorted)
        low_val = float(np.percentile(a_sorted, low_pct))
        high_val = float(np.percentile(a_sorted, high_pct))
        keep_sorted[unassigned] = (a_sorted[unassigned] >= low_val) & (
            a_sorted[unassigned] <= high_val
        )

    keep = keep_sorted[inv_idx]
    baseline = baseline_sorted[inv_idx]
    return keep, baseline


def _symmetric_complex_derivative(freq_sorted: np.ndarray, s21_sorted: np.ndarray) -> np.ndarray:
    n = len(freq_sorted)
    if n == 0:
        return np.zeros(0, dtype=complex)
    if n == 1:
        return np.zeros(1, dtype=complex)

    grad = np.zeros(n, dtype=complex)
    df0 = freq_sorted[1] - freq_sorted[0]
    dfn = freq_sorted[-1] - freq_sorted[-2]
    grad[0] = (s21_sorted[1] - s21_sorted[0]) / (df0 if df0 != 0 else 1.0)
    grad[-1] = (s21_sorted[-1] - s21_sorted[-2]) / (dfn if dfn != 0 else 1.0)

    if n > 2:
        df = freq_sorted[2:] - freq_sorted[:-2]
        denom = np.where(df == 0, 1.0, df)
        grad[1:-1] = (s21_sorted[2:] - s21_sorted[:-2]) / denom
    return grad


def _compute_one_scan_filter(
    scan: VNAScan,
    width_ghz: float,
    step_ghz: float,
    retain_pct: float,
    center_pct: float,
    low_slope_pct: float,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, np.ndarray]:
    freq = scan.freq
    amp = scan.amplitude()
    s21 = scan.complex_s21()
    n = len(freq)
    if n == 0:
        return {
            "retained_mask": np.zeros(0, dtype=bool),
            "baseline_amplitude": np.zeros(0, dtype=float),
            "slope_survivor_mask": np.zeros(0, dtype=bool),
        }

    slope_pct = float(np.clip(low_slope_pct, 0.1, 100.0))
    retain_pct = float(np.clip(retain_pct, 0.1, 100.0))
    center_pct = float(np.clip(center_pct, 0.0, 100.0))
    low_pct = max(0.0, center_pct - 0.5 * retain_pct)
    high_pct = min(100.0, center_pct + 0.5 * retain_pct)

    sort_idx = np.argsort(freq)
    inv_idx = np.empty_like(sort_idx)
    inv_idx[sort_idx] = np.arange(n)
    f_sorted = freq[sort_idx]
    a_sorted = amp[sort_idx]
    s_sorted = s21[sort_idx]

    width = _window_width_in_freq_units(freq, width_ghz)
    step = _window_width_in_freq_units(freq, step_ghz)
    if width <= 0 or step <= 0:
        raise ValueError("Window width and compute step must be > 0")

    grad_sorted = _symmetric_complex_derivative(f_sorted, s_sorted)
    slope_mag_sorted = np.abs(grad_sorted)

    baseline_sorted = np.full(n, np.nan, dtype=float)
    keep_sorted = np.zeros(n, dtype=bool)
    slope_survivor_sorted = np.zeros(n, dtype=bool)

    total_span = float(f_sorted[-1] - f_sorted[0]) if n > 1 else 0.0
    n_sections = max(1, int(np.floor(total_span / step)) + 1)
    half_width = 0.5 * width
    half_step = 0.5 * step

    for k in range(n_sections):
        center = float(f_sorted[0]) + k * step
        w_left = np.searchsorted(f_sorted, center - half_width, side="left")
        w_right = np.searchsorted(f_sorted, center + half_width, side="right")
        if w_right <= w_left:
            if progress_cb is not None:
                progress_cb(k + 1, n_sections)
            continue

        w_amp = a_sorted[w_left:w_right]
        w_slope = slope_mag_sorted[w_left:w_right]
        slope_thresh = float(np.percentile(w_slope, slope_pct))
        w_slope_ok = w_slope <= slope_thresh

        # Median/value thresholds are computed only from points surviving local slope cut.
        if np.any(w_slope_ok):
            vals_for_stats = w_amp[w_slope_ok]
        else:
            vals_for_stats = w_amp
        med = float(np.median(vals_for_stats))
        low_val = float(np.percentile(vals_for_stats, low_pct))
        high_val = float(np.percentile(vals_for_stats, high_pct))

        a_left = np.searchsorted(f_sorted, center - half_step, side="left")
        a_right = np.searchsorted(f_sorted, center + half_step, side="right")
        if a_right > a_left:
            assign_amp = a_sorted[a_left:a_right]
            assign_slope = slope_mag_sorted[a_left:a_right]
            assign_slope_ok = assign_slope <= slope_thresh
            slope_survivor_sorted[a_left:a_right] = assign_slope_ok
            baseline_sorted[a_left:a_right] = med
            keep_sorted[a_left:a_right] = assign_slope_ok & (assign_amp >= low_val) & (
                assign_amp <= high_val
            )

        if progress_cb is not None:
            progress_cb(k + 1, n_sections)

    # Fill never-assigned points conservatively.
    unassigned = np.isnan(baseline_sorted)
    if np.any(unassigned):
        baseline_sorted[unassigned] = np.median(a_sorted)
        slope_survivor_sorted[unassigned] = False
        keep_sorted[unassigned] = False

    return {
        "retained_mask": keep_sorted[inv_idx],
        "baseline_amplitude": baseline_sorted[inv_idx],
        "slope_survivor_mask": slope_survivor_sorted[inv_idx],
    }


def _estimate_frequency_resolution_mhz(scans: List[VNAScan]) -> float:
    resolutions_hz: List[float] = []
    for scan in scans:
        freq = np.asarray(scan.freq, dtype=float)
        if freq.size < 2:
            continue
        diffs = np.diff(np.sort(freq))
        diffs = np.abs(diffs[np.isfinite(diffs)])
        diffs = diffs[diffs > 0]
        if diffs.size == 0:
            continue

        median_abs = float(np.nanmedian(np.abs(freq)))
        min_step = float(np.min(diffs))
        # Convert to Hz estimate based on likely input units.
        if median_abs > 1e6:
            step_hz = min_step
        elif median_abs > 1e3:
            step_hz = min_step * 1e6  # MHz -> Hz
        else:
            step_hz = min_step * 1e9  # GHz -> Hz
        resolutions_hz.append(step_hz)

    if not resolutions_hz:
        return 0.0015  # fallback: 1500 Hz
    return max(min(resolutions_hz) / 1e6, 1e-6)
