"""
Generate and plot S21 around a 1 GHz resonance for typical Al KID parameters.

This is a lightweight sanity-check script for the resonator model implementation.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from ComplexResonance import ComplexResonanceDirect


def main(show_plot=False):
    # Typical-order parameters for an Al KID resonator.
    fr = 1.0e9  # Resonance frequency [Hz]
    Qi = 3.0e5  # Internal quality factor
    Qc_mag = 1.2e5  # Magnitude of coupling Q
    phi_deg = 8.0  # Asymmetry phase [deg], modest mismatch/cable effects
    phi = np.deg2rad(phi_deg)

    # Complex coupling Q used in the Khalil-style model.
    Qcom = Qc_mag * np.exp(-1j * phi)

    # Loaded Q consistent with 1/Q = 1/Qi + Re(1/Qcom).
    Q = 1.0 / (1.0 / Qi + np.real(1.0 / Qcom))

    # Complex baseline gain and cable delay.
    a = 0.95 * np.exp(1j * np.deg2rad(10.0))
    tau = 35e-9  # 35 ns electrical delay

    # Frequency sweep: +/- 5 linewidths around resonance.
    linewidth = fr / Q
    span = 10.0 * linewidth
    npts = 4001
    f = np.linspace(fr - span / 2.0, fr + span / 2.0, npts)

    s21 = ComplexResonanceDirect(f, fr, Q, Qcom, a, tau)

    # Derived plotting quantities.
    mag_db = 20.0 * np.log10(np.abs(s21))
    phase_deg = np.unwrap(np.angle(s21)) * 180.0 / np.pi
    detuning_khz = (f - fr) / 1e3
    i_res = int(np.argmin(np.abs(f - fr)))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(detuning_khz, mag_db, lw=1.5)
    axes[0].plot(detuning_khz[i_res], mag_db[i_res], "rD", ms=7)
    axes[0].set_title("|S21|")
    axes[0].set_xlabel("Detuning from fr [kHz]")
    axes[0].set_ylabel("Magnitude [dB]")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(detuning_khz, phase_deg, lw=1.5)
    axes[1].plot(detuning_khz[i_res], phase_deg[i_res], "rD", ms=7)
    axes[1].set_title("Phase(S21)")
    axes[1].set_xlabel("Detuning from fr [kHz]")
    axes[1].set_ylabel("Phase [deg]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(np.real(s21), np.imag(s21), lw=1.5)
    axes[2].plot(np.real(s21[i_res]), np.imag(s21[i_res]), "rD", ms=7)
    axes[2].set_title("IQ Circle")
    axes[2].set_xlabel("Re(S21)")
    axes[2].set_ylabel("Im(S21)")
    axes[2].grid(True, alpha=0.3)
    axes[2].axis("equal")

    fig.suptitle(
        (
            f"Al KID-like Resonator @ 1 GHz: Qi={Qi:.0f}, |Qcom|={Qc_mag:.0f}, "
            f"Q={Q:.0f}, phi={phi_deg:.1f} deg, tau={tau*1e9:.1f} ns"
        ),
        fontsize=11,
    )
    fig.tight_layout()

    out_png = "s21_al_kid_1ghz.png"
    fig.savefig(out_png, dpi=150)
    print(f"Saved plot: {out_png}")
    print(f"fr={fr:.3e} Hz, Qi={Qi:.3e}, |Qcom|={Qc_mag:.3e}, Q_loaded={Q:.3e}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Al KID-like S21 around 1 GHz.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure window (default is save-only, no blocking show).",
    )
    args = parser.parse_args()
    main(show_plot=args.show)
