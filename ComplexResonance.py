import numpy as np

from resonance import resonance


def ComplexResonance(f, fr, Q, Qcom, a, tau):
    """
    Evaluate complex resonator transmission using a coupling-Q wrapper.

    This helper maps a more physical parameterization (`Qcom`, `a`) into the
    parameterization expected by `resonance(...)` and returns complex S21.
    In other words, this wrapper accepts complex inputs and converts them into
    the mostly real-valued magnitude/phase parameters used by `resonance(...)`.

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
    # Q is the resonator total Q
    # Qcom is the complex valued coupling Q.  (Qe~ from Khalil)

    # This function serves as a wrapper for resonance
    # it takes normal parameters and translates them in to the parameters I assigned
    # resonance function which the the core of my fitting routines
    #
    # Rat= Q/abs(Qcom) phi=arg(Q/Qcom)=-arg(Qcom) phi is not the phase of Qcom, but the phase of 1/Qcom
    # according to Khalil eqn 11 and 12
    # assume function of form Gao's theses, Khalil paper

    a_amp = np.abs(a)
    a_arg = np.angle(a)
    Rat = np.abs(Q / Qcom)
    phi = np.angle(Q / Qcom)

    # Convert to magnitude/phase form used by the core model.
    a_parm = a_amp * np.exp(1j * a_arg)
    S21 = resonance(f, fr, Q, Rat, a_amp, a_arg, tau, phi)
    return S21


def ComplexResonanceDirect(f, fr, Q, Qcom, a, tau):
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
