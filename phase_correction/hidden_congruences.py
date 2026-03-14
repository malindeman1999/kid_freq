import math
import random
from collections import Counter


def divisor_candidates(g, min_m=100, max_divisors=200):
    """
    Return divisors of g >= min_m.
    For large g this can be expensive, so cap the work.
    """
    divs = set()
    limit = int(math.isqrt(g))
    checked = 0

    for d in range(1, limit + 1):
        if g % d == 0:
            q = g // d
            if d >= min_m:
                divs.add(d)
            if q >= min_m:
                divs.add(q)
        checked += 1
        if checked > max_divisors and len(divs) > 20:
            break

    return sorted(divs)


def score_modulus(nums, m):
    """
    Score modulus m by finding the most populated residue class.
    """
    residues = Counter(x % m for x in nums)
    a, count = residues.most_common(1)[0]
    frac = count / len(nums)
    return {
        "m": m,
        "a": a,
        "count": count,
        "fraction": frac,
    }


def detect_modular_pattern(
    nums,
    trials=20000,
    min_m=100,
    top_k=20,
    random_seed=0,
    min_count_per_residue=1,
    max_residues_per_mod=10,
):
    """
    Detect x ≡ a (mod m) pattern in a noisy integer list.

    Parameters
    ----------
    nums : list[int]
        Input integers.
    trials : int
        Number of random triple samples.
    min_m : int
        Ignore moduli smaller than this.
    top_k : int
        Return best top_k candidates.
    """
    rng = random.Random(random_seed)
    nums = [int(x) for x in nums]
    n = len(nums)

    if n < 3:
        return []

    gcd_votes = Counter()

    # Stage 1: random triple sampling
    for _ in range(trials):
        i, j, k = rng.sample(range(n), 3)
        x0, x1, x2 = nums[i], nums[j], nums[k]

        d1 = abs(x1 - x0)
        d2 = abs(x2 - x0)

        if d1 == 0 and d2 == 0:
            continue

        g = math.gcd(d1, d2)
        if g >= min_m:
            gcd_votes[g] += 1

    if not gcd_votes:
        return []

    # Stage 2: expand promising gcds into divisor candidates
    candidate_votes = Counter()
    for g, vote in gcd_votes.most_common(50):
        for d in divisor_candidates(g, min_m=min_m):
            candidate_votes[d] += vote

    # Stage 3: score top candidates by residue concentration.
    # Keep multiple residues per modulus (not just the top one).
    results = []
    for m, _ in candidate_votes.most_common(200):
        residues = Counter(x % m for x in nums)
        ordered = residues.most_common()
        kept = 0
        for a, count in ordered:
            if count < min_count_per_residue:
                continue
            frac = count / n
            results.append(
                {
                    "m": m,
                    "a": int(a),
                    "count": int(count),
                    "fraction": float(frac),
                }
            )
            kept += 1
            if max_residues_per_mod is not None and kept >= max_residues_per_mod:
                break

    # Deduplicate by (m, a) and sort by support.
    seen = set()
    unique = []
    for r in sorted(results, key=lambda z: (-z["count"], -z["fraction"], -z["m"], z["a"])):
        key = (r["m"], r["a"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique[:top_k]


def label_inliers(nums, m, a):
    """
    Return which numbers fit x ≡ a (mod m).
    """
    inliers = [x for x in nums if x % m == a]
    outliers = [x for x in nums if x % m != a]
    return inliers, outliers
