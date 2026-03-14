import math

import numpy as np

from .hidden_congruences import detect_modular_pattern


def _nearest_vna_indices(freqs, vna_freqs):
    vna = np.asarray(vna_freqs, dtype=float).ravel()
    f = np.asarray(freqs, dtype=float).ravel()
    d = np.diff(vna)
    if d.size == 0:
        raise ValueError("Need at least 2 VNA frequencies.")
    df = float(np.median(d))
    tol = max(abs(df) * 1e-3, 1e-6)

    sort_idx = np.argsort(vna)
    vna_sorted = vna[sort_idx]
    pos = np.searchsorted(vna_sorted, f)
    left = np.clip(pos - 1, 0, vna_sorted.size - 1)
    right = np.clip(pos, 0, vna_sorted.size - 1)
    left_dist = np.abs(f - vna_sorted[left])
    right_dist = np.abs(f - vna_sorted[right])
    use_right = right_dist < left_dist
    chosen = np.where(use_right, right, left)
    dist = np.where(use_right, right_dist, left_dist)
    if np.any(dist > tol):
        bad = f[dist > tol][0]
        raise ValueError(f"Correction frequency {bad:.12g} not found on VNA grid.")
    return sort_idx[chosen]


def _binomial_tail_geq(n, k, p):
    """Compute P[X >= k] for X ~ Binomial(n, p)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0 if k <= n else 0.0

    log_terms = []
    log_p = math.log(p)
    log_q = math.log(1.0 - p)
    for x in range(k, n + 1):
        log_c = math.lgamma(n + 1) - math.lgamma(x + 1) - math.lgamma(n - x + 1)
        log_terms.append(log_c + x * log_p + (n - x) * log_q)

    max_log = max(log_terms)
    s = sum(math.exp(t - max_log) for t in log_terms)
    return float(math.exp(max_log) * s)


def classify_congruent_corrections(
    irregular_freqs,
    vna_freqs,
    corrected_phase_deg,
    min_separation_hz=15e3,
    p_random_cutoff=1e-3,
    verbose=False,
):
    irregular = np.asarray(irregular_freqs, dtype=float).ravel()
    vna = np.asarray(vna_freqs, dtype=float).ravel()
    phase = np.asarray(corrected_phase_deg, dtype=float).ravel()
    if verbose:
        print(
            f"[congruence] Input sizes: irregular={irregular.size}, vna={vna.size}, phase={phase.size}"
        )
    if vna.size != phase.size:
        raise ValueError("vna_freqs and corrected_phase_deg must have the same length.")

    if irregular.size == 0:
        return [], [], []

    irregular_sorted = np.sort(np.unique(irregular))
    if irregular_sorted.size == 1:
        isolated = irregular_sorted.copy()
    else:
        gaps = np.diff(irregular_sorted)
        left_gap = np.empty_like(irregular_sorted)
        right_gap = np.empty_like(irregular_sorted)
        left_gap[0] = np.inf
        left_gap[1:] = gaps
        right_gap[-1] = np.inf
        right_gap[:-1] = gaps
        nearest_gap = np.minimum(left_gap, right_gap)
        isolated = irregular_sorted[nearest_gap >= min_separation_hz]

    rejected = irregular_sorted[~np.isin(irregular_sorted, isolated)]

    if isolated.size == 0:
        return [], irregular_sorted.tolist(), rejected.tolist()

    d = np.diff(vna)
    if d.size == 0:
        raise ValueError("Need at least 2 VNA frequencies.")
    df = float(np.median(d))
    f0 = float(vna[0])
    ints = np.rint((isolated - f0) / df).astype(int)

    ref_int = int(ints[0])
    delta_ints = (ints - ref_int).astype(int)
    results = detect_modular_pattern(delta_ints.tolist(), top_k=200)
    if not results:
        congruent = np.array([], dtype=float)
        non_congruent = irregular_sorted
    else:
        n_pts = int(delta_ints.size)
        for cand in results:
            m = max(1, int(cand["m"]))
            c = int(cand["count"])
            p_base = 1.0 / m
            p_single = _binomial_tail_geq(n_pts, c, p_base)
            cand["p_random"] = min(1.0, m * p_single)

        selected = [
            cand for cand in results if float(cand.get("p_random", 1.0)) < p_random_cutoff
        ]

        all_ints = np.rint((irregular_sorted - f0) / df).astype(int)
        all_delta_ints = (all_ints - ref_int).astype(int)
        all_mask = np.zeros(all_delta_ints.shape, dtype=bool)
        for cand in selected:
            m, a = int(cand["m"]), int(cand["a"])
            all_mask |= (all_delta_ints % m) == a

        congruent = irregular_sorted[all_mask]
        non_congruent = irregular_sorted[~all_mask]

    if verbose:
        print(
            f"[congruence] Done. congruent={congruent.size}, non_congruent={non_congruent.size}"
        )
    return congruent.tolist(), non_congruent.tolist(), rejected.tolist()
