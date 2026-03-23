import numpy as np


def ComplexResonance(f, fr, Q, Qcom, a, tau):
    """
    Evaluate complex resonator transmission directly from complex parameters.

    This version computes S21 without converting `Qcom` and `a` into separate
    magnitude/phase parameters first. It is algebraically equivalent to
    `ComplexResonance(...)` for the same inputs.

    Parameters
    ----------
    f : float or ndarray
        Frequency (Hz) at which to evaluate S21.
    fr : float
        Resonance frequency (Hz).
    Q : float
        Loaded/total resonator quality factor.
    Qcom : complex
        Complex coupling quality factor (Khalil's \\hat{Q}_e).
    a : complex
        Complex gain and static phase prefactor.
    tau : float
        Electrical delay (seconds), used in exp(i*2*pi*f*tau).

    Returns
    -------
    complex or ndarray
        Complex transmission S21 evaluated at `f`.
    """
    # Direct complex form of the same resonator model used by resonance(...).
    coupling_term = Q / Qcom
    detuning_term = 1 + 1j * 2 * Q * ((f - fr) / fr)
    S21 = a * np.exp(1j * 2 * np.pi * f * tau) * (1 - coupling_term / detuning_term)
    return S21


def ComplexResonanceQi(f, fr, Qi, Qcom, a, tau):
    """
    Evaluate complex resonator transmission using internal Qi and complex coupling Q.

    This wrapper converts the physically meaningful internal quality factor `Qi`
    and complex coupling quality factor `Qcom` into the loaded quality factor
    expected by `ComplexResonance(...)`.
    """
    q_loaded = 1.0 / (1.0 / Qi + np.real(1.0 / Qcom))
    return ComplexResonance(f, fr, q_loaded, Qcom, a, tau)
