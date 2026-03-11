# Parameterization Difference: `ComplexResonance` vs `resonance`

This project has two related functions that use the same resonator model, but with different input parameterizations.

## High-Level Summary

- `resonance(...)` is the core model.
- `ComplexResonance(...)` is a wrapper that converts physically intuitive complex parameters into the core model parameters.

## Function Signatures

```python
ComplexResonance(f, fr, Q, Qcom, a, tau)
resonance(f, fr, Q, Rat, a_amp, a_arg, tau, phi)
```

## Parameter Mapping

`ComplexResonance` computes:

- `a_amp = abs(a)`
- `a_arg = angle(a)`
- `Rat = abs(Q / Qcom)`  (equivalently `Q / abs(Qcom)` when `Q > 0`)
- `phi = angle(Q / Qcom)` (equivalently `-angle(Qcom)` when `Q` is real and positive)

Then it calls:

```python
resonance(f, fr, Q, Rat, a_amp, a_arg, tau, phi)
```

## Interpretation of Each Style

### `ComplexResonance` parameterization (wrapper)

- `Qcom` (complex): complex coupling quality factor (`Qe_hat` style parameter)
- `a` (complex): combined amplitude + static phase term
- `tau` (real): electrical delay (seconds)

This form is convenient when your fit variables are naturally complex (`Qcom`, `a`).

### `resonance` parameterization (core)

- `Rat` (real): coupling magnitude factor (`Q/|Qcom|`)
- `phi` (real): asymmetry phase (`arg(Q/Qcom)`)
- `a_amp` (real): magnitude of prefactor
- `a_arg` (real): phase of prefactor
- `tau` (real): electrical delay (seconds)

This form is convenient when you want explicit control of magnitude/phase components.

## Same Underlying Model

Both functions represent the same complex transmission model:

```python
S21 = a_amp * exp(i*a_arg) * exp(i*2*pi*f*tau) * (
    1 - Rat * exp(i*phi) / (1 + i*2*Q*((f-fr)/fr))
)
```

So the difference is not physics; it is only how the inputs are parameterized before evaluating the same equation.
