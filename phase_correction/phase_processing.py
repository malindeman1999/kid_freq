import numpy as np

from .congruence_classifier import classify_congruent_corrections
from .phase_correction import correct_phase_diffs


def _nearest_indices(query_freqs, ref_freqs):
    q = np.asarray(query_freqs, dtype=float).ravel()
    if q.size == 0:
        return np.empty((0,), dtype=int)
    ref = np.asarray(ref_freqs, dtype=float).ravel()
    d = np.diff(ref)
    df = float(np.median(d)) if d.size > 0 else 1.0
    tol = max(abs(df) * 1e-3, 1e-6)

    sort_idx = np.argsort(ref)
    ref_sorted = ref[sort_idx]
    pos = np.searchsorted(ref_sorted, q)
    left = np.clip(pos - 1, 0, ref_sorted.size - 1)
    right = np.clip(pos, 0, ref_sorted.size - 1)
    left_dist = np.abs(q - ref_sorted[left])
    right_dist = np.abs(q - ref_sorted[right])
    use_right = right_dist < left_dist
    chosen = np.where(use_right, right, left)
    dist = np.where(use_right, right_dist, left_dist)
    if np.any(dist > tol):
        bad = q[dist > tol][0]
        raise ValueError(f"Frequency {bad:.12g} not found on reference grid.")
    return sort_idx[chosen]


def _nearest_phase_values(query_freqs, ref_freqs, phase_values):
    idx = _nearest_indices(query_freqs, ref_freqs)
    phase = np.asarray(phase_values, dtype=float).ravel()
    return phase[idx]


def process_phase_data(
    freq,
    complex_data,
    threshold_deg=10.0,
    apply_exact_360=True,
    max_passes=3,
    min_separation_hz=15e3,
    p_random_cutoff=1e-3,
    verbose=False,
):
    freq = np.asarray(freq, dtype=float).ravel()
    complex_data = np.asarray(complex_data, dtype=complex).ravel()

    real = np.real(complex_data)
    imag = np.imag(complex_data)
    magnitude = np.abs(complex_data)
    phase = np.angle(real + 1j * imag, deg=True)

    (
        phase_corrected,
        _,
        correction_360_freqs,
        _,
        correction_irregular_freqs,
        _,
    ) = correct_phase_diffs(
        phase,
        freq=freq,
        threshold_deg=threshold_deg,
        apply_exact_360=apply_exact_360,
        max_passes=max_passes,
        return_details=True,
        verbose=verbose,
    )
    phase_corrected_initial = phase_corrected.copy()

    congruent_freqs, non_congruent_freqs, _ = classify_congruent_corrections(
        correction_irregular_freqs,
        freq,
        phase_corrected,
        min_separation_hz=min_separation_hz,
        p_random_cutoff=p_random_cutoff,
        verbose=verbose,
    )

    all_irregular_idx = set(_nearest_indices(correction_irregular_freqs, freq).tolist())
    congruent_idx = set(_nearest_indices(congruent_freqs, freq).tolist())
    non_congruent_idx = set(_nearest_indices(non_congruent_freqs, freq).tolist())
    overlap = congruent_idx & non_congruent_idx
    if overlap:
        raise RuntimeError(
            f"Invalid classification: {len(overlap)} irregular points are both congruent and non-congruent."
        )
    classified_idx = congruent_idx | non_congruent_idx
    missing_idx = all_irregular_idx - classified_idx
    extra_idx = classified_idx - all_irregular_idx
    if missing_idx or extra_idx:
        raise RuntimeError("Invalid classification: irregular points are not exhaustively classified.")

    phase_diff_original = np.diff(phase)
    phase_diff_corrected = np.diff(phase_corrected)
    non_congruent_idx_arr = _nearest_indices(non_congruent_freqs, freq)
    non_congruent_diff_idx = non_congruent_idx_arr - 1
    non_congruent_diff_idx = non_congruent_diff_idx[non_congruent_diff_idx >= 0]
    phase_diff_corrected[non_congruent_diff_idx] = phase_diff_original[non_congruent_diff_idx]
    phase_corrected[1:] = phase_corrected[0] + np.cumsum(phase_diff_corrected)

    return {
        "real": real,
        "imag": imag,
        "magnitude": magnitude,
        "phase": phase,
        "phase_corrected": phase_corrected,
        "phase_corrected_initial": phase_corrected_initial,
        "phase_corrected_initial_mod360": np.mod(phase_corrected_initial, 360.0),
        "phase_corrected_mod360": np.mod(phase_corrected, 360.0),
        "phase_diff": phase_diff_original,
        "phase_corrected_diff": phase_diff_corrected,
        "freq_diff": freq[1:],
        "correction_360_freqs": correction_360_freqs,
        "correction_360_phases_mod360": np.mod(
            _nearest_phase_values(correction_360_freqs, freq, phase_corrected), 360.0
        ),
        "correction_irregular_freqs": correction_irregular_freqs,
        "congruent_freqs": np.asarray(congruent_freqs, dtype=float),
        "congruent_phases_mod360": np.mod(
            _nearest_phase_values(congruent_freqs, freq, phase_corrected), 360.0
        ),
        "non_congruent_freqs": np.asarray(non_congruent_freqs, dtype=float),
        "non_congruent_phases_mod360": np.mod(
            _nearest_phase_values(non_congruent_freqs, freq, phase_corrected), 360.0
        ),
    }
