import warnings

import numpy as np


def _find_expected_diff(source_diffs, i, threshold_deg):
    left_idx = None
    for j in range(i - 1, -1, -1):
        if abs(source_diffs[j]) <= threshold_deg:
            left_idx = j
            break

    right_idx = None
    for j in range(i + 1, source_diffs.size):
        if abs(source_diffs[j]) <= threshold_deg:
            right_idx = j
            break

    if left_idx is None or right_idx is None:
        return 0.0

    left_val = float(source_diffs[left_idx])
    right_val = float(source_diffs[right_idx])
    span = right_idx - left_idx
    if span <= 0:
        return 0.0

    t = (i - left_idx) / span
    return left_val + t * (right_val - left_val)


def _snap_correction_to_360(correction_value, threshold_deg):
    k = round(correction_value / 360.0)
    if k == 0:
        return correction_value, False
    snapped = k * 360.0
    if abs(correction_value - snapped) <= threshold_deg:
        return snapped, True
    return correction_value, False


def _single_pass(phase, threshold_deg, apply_exact_360):
    source_diffs = np.diff(phase)
    corrected_diffs = source_diffs.copy()
    correction_applied = np.zeros_like(source_diffs, dtype=float)
    corrected_mask = np.zeros_like(source_diffs, dtype=bool)
    correction_360_mask = np.zeros_like(source_diffs, dtype=bool)
    correction_irregular_mask = np.zeros_like(source_diffs, dtype=bool)

    for i in range(source_diffs.size):
        expected = _find_expected_diff(source_diffs, i, threshold_deg)
        if abs(source_diffs[i] - expected) > threshold_deg:
            correction_value = expected - source_diffs[i]
            used_360_snap = False
            if apply_exact_360:
                correction_value, used_360_snap = _snap_correction_to_360(
                    correction_value, threshold_deg
                )
            correction_applied[i] = correction_value
            corrected_diffs[i] = source_diffs[i] + correction_value
            corrected_mask[i] = True
            if used_360_snap:
                correction_360_mask[i] = True
            else:
                correction_irregular_mask[i] = True

    corrected_phase = np.empty_like(phase, dtype=float)
    corrected_phase[0] = phase[0]
    corrected_phase[1:] = phase[0] + np.cumsum(corrected_diffs)
    return (
        corrected_phase,
        correction_applied,
        corrected_mask,
        correction_360_mask,
        correction_irregular_mask,
    )


def _remaining_exceed_count(phase, threshold_deg):
    diffs = np.diff(phase)
    count = 0
    for i in range(diffs.size):
        expected = _find_expected_diff(diffs, i, threshold_deg)
        if abs(diffs[i] - expected) > threshold_deg:
            count += 1
    return count


def correct_phase_diffs(
    phase_deg,
    freq=None,
    threshold_deg=10.0,
    apply_exact_360=True,
    max_passes=3,
    return_details=False,
    verbose=False,
):
    phase = np.asarray(phase_deg, dtype=float).ravel()
    freq_arr = None
    if freq is not None:
        freq_arr = np.asarray(freq, dtype=float).ravel()
        if freq_arr.size != phase.size:
            raise ValueError(
                f"freq length ({freq_arr.size}) must match phase length ({phase.size})."
            )

    if phase.size <= 1:
        if return_details:
            return (
                phase.copy(),
                np.empty((0, 0), dtype=float),
                np.empty((0,), dtype=float),
                np.empty((0,), dtype=float),
                np.empty((0,), dtype=float),
                np.empty((0,), dtype=float),
            )
        return phase.copy()

    corrected_phase = phase.copy()
    correction_history = []
    correction_360_freqs_list = []
    correction_360_phases_list = []
    correction_irregular_freqs_list = []
    correction_irregular_phases_list = []

    for pass_idx in range(max(1, int(max_passes))):
        (
            corrected_phase,
            correction_applied,
            corrected_mask,
            correction_360_mask,
            correction_irregular_mask,
        ) = _single_pass(corrected_phase, threshold_deg, apply_exact_360)
        correction_history.append(correction_applied)

        if freq_arr is not None and np.any(corrected_mask):
            if np.any(correction_360_mask):
                idx_360 = np.flatnonzero(correction_360_mask) + 1
                correction_360_freqs_list.append(freq_arr[idx_360])
                correction_360_phases_list.append(corrected_phase[idx_360])
            if np.any(correction_irregular_mask):
                idx_irregular = np.flatnonzero(correction_irregular_mask) + 1
                correction_irregular_freqs_list.append(freq_arr[idx_irregular])
                correction_irregular_phases_list.append(corrected_phase[idx_irregular])

        exceed_count = _remaining_exceed_count(corrected_phase, threshold_deg)
        if verbose:
            print(
                f"Phase correction pass {pass_idx + 1}/{max_passes}: "
                f"{exceed_count} diff(s) exceed threshold {threshold_deg:.6g} deg."
            )
        if exceed_count == 0:
            break

    exceed_count = _remaining_exceed_count(corrected_phase, threshold_deg)
    if exceed_count > 0:
        warnings.warn(
            f"Phase correction incomplete after {max_passes} passes: "
            f"{exceed_count} diff(s) still exceed threshold {threshold_deg:.6g} deg.",
            RuntimeWarning,
            stacklevel=2,
        )

    if return_details:
        correction_360_freqs = (
            np.concatenate(correction_360_freqs_list)
            if correction_360_freqs_list
            else np.empty((0,), dtype=float)
        )
        correction_360_phases = (
            np.concatenate(correction_360_phases_list)
            if correction_360_phases_list
            else np.empty((0,), dtype=float)
        )
        correction_irregular_freqs = (
            np.concatenate(correction_irregular_freqs_list)
            if correction_irregular_freqs_list
            else np.empty((0,), dtype=float)
        )
        correction_irregular_phases = (
            np.concatenate(correction_irregular_phases_list)
            if correction_irregular_phases_list
            else np.empty((0,), dtype=float)
        )
        return (
            corrected_phase,
            np.asarray(correction_history),
            correction_360_freqs,
            correction_360_phases,
            correction_irregular_freqs,
            correction_irregular_phases,
        )
    return corrected_phase
