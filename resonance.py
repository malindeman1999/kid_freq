import numpy as np


def resonance(f, fr, Q, Rat, a_amp, a_arg, tau, phi):
    """
    Compute the complex resonator transmission S21(f).

    Model form follows the asymmetric resonator expression used in
    Khalil et al. (J. Appl. Phys. 111, 054510 (2012), Eqs. 11-12), with an
    additional complex prefactor and cable-delay phase term.

    Note
    ----
    This function is parameterized mostly by real-valued inputs
    (`fr`, `Q`, `Rat`, `a_amp`, `a_arg`, `tau`, `phi`, and usually `f`),
    but it computes a complex-valued transmission `S21`.

    Parameters
    ----------
    f : float or ndarray
        Frequency (Hz).
    fr : float
        Resonance frequency (Hz).
    Q : float
        Loaded/total quality factor.
    Rat : float
        Q/|Qcom|, where Qcom is the complex coupling Q.
    a_amp : float
        Magnitude of complex gain prefactor.
    a_arg : float
        Phase (radians) of complex gain prefactor.
    tau : float
        Electrical delay (seconds).
    phi : float
        Asymmetry phase (radians), arg(Q/Qcom).

    Returns
    -------
    complex or ndarray
        Complex transmission S21 (named `Z` here).
    """
    # Rat= Q/abs(Qcom) phi=arg(Q/Qcom)=-arg(Qcom) phi is not the phase of Qcom, but the phase of 1/Qcom
    # according to Khalil eqn 11 and 12
    # assume function of form Gao's theses, Khalil paper
    # Overall complex gain/phase prefactor.
    a_parm = a_amp * np.exp(1j * a_arg)
    # Resonator response with cable delay and asymmetric complex coupling term.
    Z = a_parm * np.exp(1j * 2 * np.pi * f * tau) * (1 - Rat * np.exp(1j * phi) / (1 + 1j * 2 * Q * ((f - fr) / fr)))
    return Z
