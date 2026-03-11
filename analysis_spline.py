from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from analysis_models import VNAScan


def fit_complex_spline_for_scan(
    scan: VNAScan, smoothing_factor: Optional[float] = None
) -> Dict[str, np.ndarray]:
    try:
        from scipy.interpolate import UnivariateSpline
    except Exception as exc:
        raise RuntimeError("scipy is required for spline fitting.") from exc

    baseline = scan.baseline_filter
    if not isinstance(baseline, dict):
        raise ValueError("Scan has no attached baseline filter data.")

    filtered_data = baseline.get("filtered_data")
    if filtered_data is None:
        raise ValueError("Attached baseline filter is missing 'filtered_data'.")

    arr = np.asarray(filtered_data)
    if arr.ndim != 2 or arr.shape[0] < 3:
        raise ValueError(f"Expected filtered_data shape (3, N), got {arr.shape}")
    if arr.shape[1] < 4:
        raise ValueError("Need at least 4 filtered points for cubic spline fit.")

    f = arr[0, :]
    amp = arr[1, :]
    phase_deg = arr[2, :]
    order = np.argsort(f)
    f = f[order]
    amp = amp[order]
    phase_deg = phase_deg[order]

    # Build smooth model by fitting amplitude and unwrapped phase directly.
    phase_rad = np.radians(phase_deg)

    spl_amp = UnivariateSpline(f, amp, s=smoothing_factor)
    spl_phase = UnivariateSpline(f, phase_rad, s=smoothing_factor)

    f_full = scan.freq
    fit_amp = spl_amp(f_full)
    fit_phase_rad = spl_phase(f_full)
    fit_c = fit_amp * np.exp(1j * fit_phase_rad)

    d_amp = spl_amp.derivative()(f_full)
    d_phase = spl_phase.derivative()(f_full)
    dfit_c = np.exp(1j * fit_phase_rad) * (d_amp + 1j * fit_amp * d_phase)

    return {
        "fit_freq": f_full,
        "fit_real": np.real(fit_c),
        "fit_imag": np.imag(fit_c),
        "fit_complex": fit_c,
        "fit_amp": fit_amp,
        "fit_phase_deg": np.degrees(fit_phase_rad),
        "derivative_complex": dfit_c,
        "derivative_mag": np.abs(dfit_c),
        "fit_basis": "amplitude_phase",
        "smoothing_factor": smoothing_factor,
    }
