from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.optimize import least_squares
from tkinter import messagebox

from ..analysis_models import _complex_from_polar, _make_event, _read_polar_series
from resonator.ComplexResonance import ComplexResonanceQi

_HZ_PER_GHZ = 1.0e9
_CITKID_MAIN = (
    Path(__file__).resolve().parents[4]
    / "PRIMA KID RESONATOR"
    / "loganfoot"
    / "citkid-main"
)
if _CITKID_MAIN.exists() and str(_CITKID_MAIN) not in sys.path:
    sys.path.insert(0, str(_CITKID_MAIN))

try:
    from citkid.res.fitter import fit_nonlinear_iq
    from citkid.res.funcs import nonlinear_iq
    from citkid.res.guess import guess_p0_nonlinear_iq
    from citkid.res.util import calc_qc_qi
except Exception:
    fit_nonlinear_iq = None
    nonlinear_iq = None
    guess_p0_nonlinear_iq = None
    calc_qc_qi = None


class ResonanceSelectionMixin:
    def _res_update_fit_mode_controls(self) -> None:
        amp_only = self.res_fit_mode_var is not None and self.res_fit_mode_var.get() == "amplitude"
        state = tk.DISABLED if amp_only else tk.NORMAL
        if amp_only:
            if self.res_a_phase_var is not None:
                self.res_a_phase_var.set("0")
            if self.res_tau_var is not None:
                self.res_tau_var.set("0")
            if self.res_fix_a_phase_var is not None:
                self.res_fix_a_phase_var.set(True)
            if self.res_fix_tau_var is not None:
                self.res_fix_tau_var.set(True)
        if getattr(self, "res_a_phase_entry", None) is not None:
            self.res_a_phase_entry.configure(state=state)
        if getattr(self, "res_tau_entry", None) is not None:
            self.res_tau_entry.configure(state=state)
        if getattr(self, "res_fix_a_phase_check", None) is not None:
            self.res_fix_a_phase_check.configure(state=state)
        if getattr(self, "res_fix_tau_check", None) is not None:
            self.res_fix_tau_check.configure(state=state)

    def _res_set_model_fields(
        self,
        *,
        fr_hz: float,
        qi: float,
        q_cpl_mag: float,
        q_cpl_phase_deg: float,
        a_mag: float,
        a_phase_deg: float,
        tau_s: float,
    ) -> None:
        if self.res_fr_var is not None:
            self.res_fr_var.set(f"{fr_hz / _HZ_PER_GHZ:.9g}")
        if self.res_qi_var is not None:
            self.res_qi_var.set(f"{qi:.9g}")
        if self.res_qc_var is not None:
            self.res_qc_var.set(f"{q_cpl_mag:.9g}")
        if self.res_qc_phase_var is not None:
            self.res_qc_phase_var.set(f"{q_cpl_phase_deg:.9g}")
        if self.res_a_mag_var is not None:
            self.res_a_mag_var.set(f"{a_mag:.9g}")
        if self.res_a_phase_var is not None:
            self.res_a_phase_var.set(f"{a_phase_deg:.9g}")
        if self.res_tau_var is not None:
            self.res_tau_var.set(f"{tau_s:.9g}")

    def _res_get_model_params_from_fields(self, *, lo: float, hi: float) -> tuple[float, float, complex, complex, float]:
        fr_hz = float(self.res_fr_var.get()) * _HZ_PER_GHZ if self.res_fr_var is not None else 0.5 * (lo + hi)
        qi = float(self.res_qi_var.get()) if self.res_qi_var is not None else 1.0e9
        q_cpl_mag = float(self.res_qc_var.get()) if self.res_qc_var is not None else 1.0e4
        q_cpl_phase_deg = float(self.res_qc_phase_var.get()) if self.res_qc_phase_var is not None else 0.0
        a_mag = float(self.res_a_mag_var.get()) if self.res_a_mag_var is not None else 1.0
        a_phase_deg = float(self.res_a_phase_var.get()) if self.res_a_phase_var is not None else 0.0
        tau_s = float(self.res_tau_var.get()) if self.res_tau_var is not None else 0.0
        qcom = q_cpl_mag * np.exp(1j * np.deg2rad(q_cpl_phase_deg))
        a = a_mag * np.exp(1j * np.deg2rad(a_phase_deg))
        return fr_hz, qi, qcom, a, tau_s

    def _res_fit_fix_flags(self) -> dict[str, bool]:
        return {
            "fr": bool(self.res_fix_fr_var.get()) if self.res_fix_fr_var is not None else False,
            "qi": bool(self.res_fix_qi_var.get()) if self.res_fix_qi_var is not None else False,
            "qc": bool(self.res_fix_qc_var.get()) if self.res_fix_qc_var is not None else False,
            "qc_phase": bool(self.res_fix_qc_phase_var.get()) if self.res_fix_qc_phase_var is not None else False,
            "a_mag": bool(self.res_fix_a_mag_var.get()) if self.res_fix_a_mag_var is not None else False,
            "a_phase": bool(self.res_fix_a_phase_var.get()) if self.res_fix_a_phase_var is not None else False,
            "tau": bool(self.res_fix_tau_var.get()) if self.res_fix_tau_var is not None else False,
        }

    def _res_logan_downward(self) -> bool:
        var = getattr(self, "res_fix_a_mag_var", None)
        return bool(var.get()) if var is not None else True

    def _res_set_logan_model_fields(self, params: np.ndarray) -> None:
        p = np.asarray(params, dtype=float)
        if p.size < 8:
            return
        if self.res_fr_var is not None:
            self.res_fr_var.set(f"{p[0] / _HZ_PER_GHZ:.9g}")
        if self.res_qi_var is not None:
            self.res_qi_var.set(f"{p[1]:.9g}")
        if self.res_qc_var is not None:
            self.res_qc_var.set(f"{p[2]:.9g}")
        if self.res_qc_phase_var is not None:
            self.res_qc_phase_var.set(f"{np.degrees(p[3]):.9g}")
        if self.res_a_mag_var is not None:
            self.res_a_mag_var.set(f"{p[4]:.9g}")
        if self.res_a_phase_var is not None:
            self.res_a_phase_var.set(f"{p[5]:.9g}")
        if getattr(self, "res_q0_var", None) is not None:
            self.res_q0_var.set(f"{p[6]:.9g}")
        if self.res_tau_var is not None:
            self.res_tau_var.set(f"{p[7]:.9g}")

    def _res_set_logan_output_fields(self, payload: dict | None) -> None:
        true_var = getattr(self, "res_true_fr_var", None)
        delta_var = getattr(self, "res_delta_fr_var", None)
        nrmse_var = getattr(self, "res_nrmse_var", None)
        qi_var = getattr(self, "res_qi_output_var", None)
        qc_var = getattr(self, "res_qc_output_var", None)
        if payload is None:
            if true_var is not None:
                true_var.set("")
            if delta_var is not None:
                delta_var.set("")
            if nrmse_var is not None:
                nrmse_var.set("")
            if qi_var is not None:
                qi_var.set("")
            if qc_var is not None:
                qc_var.set("")
            return
        true_fr = payload.get("true_fr_hz")
        delta_fr = payload.get("delta_fr_hz")
        nrmse = payload.get("nrmse")
        qi = payload.get("q_internal")
        qc = payload.get("q_coupling")
        if true_var is not None:
            true_var.set("" if true_fr is None else f"{float(true_fr) / _HZ_PER_GHZ:.9g}")
        if delta_var is not None:
            delta_var.set("" if delta_fr is None else f"{float(delta_fr):.9g}")
        if nrmse_var is not None:
            nrmse_var.set("" if nrmse is None else f"{float(nrmse):.9g}")
        if qi_var is not None:
            qi_var.set("" if qi is None else f"{float(qi):.9g}")
        if qc_var is not None:
            qc_var.set("" if qc is None else f"{float(qc):.9g}")

    def _res_get_logan_params_from_fields(self, *, lo: float, hi: float) -> np.ndarray:
        fr_hz = float(self.res_fr_var.get()) * _HZ_PER_GHZ if self.res_fr_var is not None else 0.5 * (lo + hi)
        qr = float(self.res_qi_var.get()) if self.res_qi_var is not None else 1.0e4
        amp = float(self.res_qc_var.get()) if self.res_qc_var is not None else 0.5
        phi_rad = (
            np.radians(float(self.res_qc_phase_var.get()))
            if self.res_qc_phase_var is not None
            else 0.0
        )
        a_nl = float(self.res_a_mag_var.get()) if self.res_a_mag_var is not None else 0.0
        i0 = float(self.res_a_phase_var.get()) if self.res_a_phase_var is not None else 1.0
        q0 = float(self.res_q0_var.get()) if getattr(self, "res_q0_var", None) is not None else 0.0
        tau_s = float(self.res_tau_var.get()) if self.res_tau_var is not None else 0.0
        return np.asarray([fr_hz, qr, amp, phi_rad, a_nl, i0, q0, tau_s], dtype=float)

    def _res_logan_fields_are_filled(self) -> bool:
        fields = [
            self.res_fr_var,
            self.res_qi_var,
            self.res_qc_var,
            self.res_qc_phase_var,
            self.res_a_mag_var,
            self.res_a_phase_var,
            getattr(self, "res_q0_var", None),
            self.res_tau_var,
        ]
        return all(field is not None and str(field.get()).strip() for field in fields)

    def _res_clear_logan_params(self) -> None:
        for field in (
            self.res_fr_var,
            self.res_qi_var,
            self.res_qc_var,
            self.res_qc_phase_var,
            self.res_a_mag_var,
            self.res_a_phase_var,
            getattr(self, "res_q0_var", None),
            self.res_tau_var,
        ):
            if field is not None:
                field.set("")
        self.res_model_preview = None
        self._res_set_logan_output_fields(None)
        self._res_set_status("Cleared Logan parameters. Next fit will guess from the visible raw S21 window.", "dark green")
        self._res_render()

    def _res_guess_logan_params(self, scan, mask: np.ndarray, lo: float, hi: float) -> np.ndarray:
        freq = np.asarray(scan.freq, dtype=float)
        z = np.asarray(scan.s21_complex_raw, dtype=np.complex128)
        try:
            return np.asarray(guess_p0_nonlinear_iq(freq[mask], z[mask]), dtype=float)
        except Exception:
            fr0 = float(0.5 * (lo + hi))
            z0 = complex(np.mean(z[mask])) if np.count_nonzero(mask) else 1.0 + 0j
            return np.asarray([fr0, 1.0e4, 0.5, 0.0, 0.0, z0.real, z0.imag, 0.0], dtype=float)

    def _res_compute_logan_frequency_shift(self, *, lo: float, hi: float, popt: np.ndarray) -> dict[str, float]:
        params = np.asarray(popt, dtype=float)
        if params.size < 8:
            return {}
        fr0 = float(params[0])
        qr = float(params[1])
        a_nl = float(params[4])
        if not np.isfinite(fr0) or not np.isfinite(qr) or qr == 0.0 or not np.isfinite(a_nl):
            return {}
        # Paper convention: nonlinear kinetic inductance is a soft spring, so
        # positive a shifts the resonance downward from fr0.
        delta_fr = -fr0 * a_nl / qr
        true_fr = fr0 + delta_fr
        return {
            "true_fr_hz": float(true_fr),
            "delta_fr_hz": float(delta_fr),
            "delta_fr_over_fr0": float(delta_fr / fr0) if fr0 != 0.0 else np.nan,
        }

    def _res_make_logan_fit_payload(
        self,
        scan,
        *,
        lo: float,
        hi: float,
        p0: np.ndarray,
        popt: np.ndarray,
        perr: np.ndarray,
        nrmse: float,
        model: np.ndarray,
        f_fit: np.ndarray,
        success: bool,
        message: str,
    ) -> dict:
        qc = np.nan
        qi = np.nan
        if calc_qc_qi is not None:
            try:
                qc, qi = calc_qc_qi(float(popt[1]), float(popt[2]))
            except Exception:
                qc = np.nan
                qi = np.nan
        shift = self._res_compute_logan_frequency_shift(lo=lo, hi=hi, popt=popt)
        return {
            "scan_key": self._scan_key(scan),
            "selection_range_hz": (float(lo), float(hi)),
            "model": "citkid.res.nonlinear_iq",
            "parameter_names": ["fr0", "Qr", "amp", "phi", "a", "i0", "q0", "tau"],
            "p0": np.asarray(p0, dtype=float),
            "popt": np.asarray(popt, dtype=float),
            "perr": np.asarray(perr, dtype=float),
            "fr0_hz": float(popt[0]),
            "fr_hz": float(popt[0]),
            "q_loaded": float(popt[1]),
            "q_coupling": float(qc),
            "q_internal": float(qi),
            "amp": float(popt[2]),
            "phi_rad": float(popt[3]),
            "phi_deg": float(np.degrees(popt[3])),
            "a_nl": float(popt[4]),
            "i0": float(popt[5]),
            "q0": float(popt[6]),
            "tau_s": float(popt[7]),
            "downward": self._res_logan_downward(),
            "nrmse": float(nrmse),
            "success": bool(success),
            "message": str(message),
            "fit_freq_hz": np.asarray(f_fit, dtype=float),
            "fit_s21_complex": np.asarray(model, dtype=np.complex128),
            **shift,
        }

    def _res_build_model_preview(self, scan, *, lo: float, hi: float) -> dict:
        freq = np.asarray(scan.freq, dtype=float)
        mask = (freq >= lo) & (freq <= hi)
        f_fit = freq[mask]
        fr_hz, qi, qcom, a, tau_s = self._res_get_model_params_from_fields(lo=lo, hi=hi)
        model = ComplexResonanceQi(f_fit, fr_hz, qi, qcom, a, tau_s)
        return {
            "scan_key": self._scan_key(scan),
            "selection_range_hz": (float(lo), float(hi)),
            "fr_hz": float(fr_hz),
            "q_internal": float(qi),
            "q_loaded": float(1.0 / (1.0 / qi + np.real(1.0 / qcom))),
            "q_coupling_mag": float(np.abs(qcom)),
            "q_coupling_phase_deg": float(np.degrees(np.angle(qcom))),
            "a_mag": float(np.abs(a)),
            "a_phase_deg": float(np.degrees(np.angle(a))),
            "tau_s": float(tau_s),
            "fit_freq_hz": np.asarray(f_fit, dtype=float),
            "fit_s21_complex": np.asarray(model, dtype=np.complex128),
        }

    def _res_display_current_model(self) -> None:
        scan = self._res_get_scan()
        if scan is None:
            return
        if nonlinear_iq is None:
            self._res_set_status("citkid.res is unavailable; cannot plot the Logan model.", "dark orange")
            return
        freq = np.asarray(scan.freq, dtype=float)
        (lo, hi), mask = self._res_get_selection_mask(freq)
        if np.count_nonzero(mask) < 2:
            self._res_set_status("Need a wider displayed region to plot the model.", "dark orange")
            return
        try:
            p0 = self._res_get_logan_params_from_fields(lo=lo, hi=hi)
            f_fit = freq[mask]
            model = nonlinear_iq(
                f_fit,
                p0[0],
                p0[1],
                p0[2],
                p0[3],
                p0[4],
                p0[5],
                p0[6],
                p0[7],
                self._res_logan_downward(),
            )
            self.res_model_preview = self._res_make_logan_fit_payload(
                scan,
                lo=lo,
                hi=hi,
                p0=np.asarray(p0, dtype=float),
                popt=np.asarray(p0, dtype=float),
                perr=np.zeros(8, dtype=float),
                nrmse=np.nan,
                model=model,
                f_fit=f_fit,
                success=True,
                message="Displayed current Logan model parameters.",
            )
            self._res_set_status("Displayed current Logan model parameters.", "dark green")
            self._res_render()
        except Exception as exc:
            self._res_set_status(f"Model plot failed: {exc}", "dark orange")

    def _res_get_selection_mask(self, freq: np.ndarray) -> tuple[tuple[float, float], np.ndarray]:
        if self._res_selected_range is None:
            self._res_selected_range = (float(freq[0]), float(freq[-1]))
        fmin, fmax = self._res_selected_range
        lo, hi = (fmin, fmax) if fmin <= fmax else (fmax, fmin)
        return (lo, hi), (freq >= lo) & (freq <= hi)

    def _res_fit_initial_frequency(self, lo: float, hi: float, gfreq: np.ndarray, dfreq: np.ndarray) -> float:
        center = 0.5 * (lo + hi)
        in_range = dfreq[(dfreq >= lo) & (dfreq <= hi)]
        if in_range.size:
            return float(in_range[np.argmin(np.abs(in_range - center))])
        in_range = gfreq[(gfreq >= lo) & (gfreq <= hi)]
        if in_range.size:
            return float(in_range[np.argmin(np.abs(in_range - center))])
        return float(center)

    def _res_reset_model_parameters(self) -> None:
        scan = self._res_get_scan()
        if scan is None:
            return
        if guess_p0_nonlinear_iq is None:
            self._res_set_status("citkid.res is unavailable; cannot guess Logan parameters.", "dark orange")
            return
        freq = np.asarray(scan.freq, dtype=float)
        (lo, hi), mask = self._res_get_selection_mask(freq)
        if np.count_nonzero(mask) < 2:
            lo = float(freq[0])
            hi = float(freq[-1])
        _range, mask = self._res_get_selection_mask(freq)
        p0 = self._res_guess_logan_params(scan, mask, lo, hi)
        self._res_set_logan_model_fields(p0)
        self.res_model_preview = None
        self._res_set_status("Reset Logan parameters from the raw selected span.", "dark green")
        self._res_render()

    def _res_current_fit(self, scan) -> Optional[dict]:
        payload = scan.candidate_resonators.get("resonator_model_fit")
        if not isinstance(payload, dict):
            return None
        fit_range = payload.get("selection_range_hz")
        current = tuple(self._res_selected_range) if self._res_selected_range is not None else None
        if not isinstance(fit_range, (list, tuple)) or len(fit_range) != 2 or current is None:
            return None
        if not np.allclose(np.asarray(fit_range, dtype=float), np.asarray(current, dtype=float), rtol=0.0, atol=1e-6):
            return None
        return payload

    def _res_fit_displayed_data(self) -> None:
        scan = self._res_get_scan()
        if scan is None:
            return
        if fit_nonlinear_iq is None or nonlinear_iq is None:
            self._res_set_status(
                f"citkid.res is unavailable. Expected package path: {_CITKID_MAIN}",
                "dark orange",
            )
            return
        freq = np.asarray(scan.freq, dtype=float)
        (lo, hi), mask = self._res_get_selection_mask(freq)
        if np.count_nonzero(mask) < 8:
            self._res_set_status("Need a wider displayed region before fitting.", "dark orange")
            return

        z = np.asarray(scan.s21_complex_raw, dtype=np.complex128)
        f_fit = freq[mask]
        z_fit = z[mask]
        if self._res_logan_fields_are_filled():
            p0 = self._res_get_logan_params_from_fields(lo=lo, hi=hi)
        else:
            p0 = self._res_guess_logan_params(scan, mask, lo, hi)
            self._res_set_logan_model_fields(p0)
        fit_tau = bool(self.res_fix_tau_var.get()) if self.res_fix_tau_var is not None else True
        downward = self._res_logan_downward()

        self._res_set_busy(True, "Fitting Logan nonlinear IQ model...")
        try:
            p0_out, popt, perr, nrmse, _figax = fit_nonlinear_iq(
                f_fit,
                z_fit,
                p0=np.asarray(p0, dtype=float),
                fit_tau=fit_tau,
                downward=downward,
                plotq=False,
            )
            p0_out = np.asarray(p0_out, dtype=float)
            popt = np.asarray(popt, dtype=float)
            perr = np.asarray(perr, dtype=float)
            model = nonlinear_iq(f_fit, *popt, downward)
            fit_payload = self._res_make_logan_fit_payload(
                scan,
                lo=lo,
                hi=hi,
                p0=p0_out,
                popt=popt,
                perr=perr,
                nrmse=float(nrmse),
                model=model,
                f_fit=f_fit,
                success=True,
                message="Logan nonlinear IQ fit complete.",
            )
            scan.candidate_resonators["logan_nonlinear_iq_fit"] = fit_payload
            self.res_model_preview = fit_payload
            self._res_set_logan_model_fields(popt)
            self._res_set_logan_output_fields(fit_payload)
            scan.processing_history.append(
                _make_event(
                    "fit_logan_nonlinear_iq",
                    {
                        "selection_range_hz": [float(lo), float(hi)],
                        "fr0_hz": float(popt[0]),
                        "true_fr_hz": float(fit_payload.get("true_fr_hz", np.nan)),
                        "delta_fr_hz": float(fit_payload.get("delta_fr_hz", np.nan)),
                        "delta_fr_over_fr0": float(fit_payload.get("delta_fr_over_fr0", np.nan)),
                        "qr": float(popt[1]),
                        "amp": float(popt[2]),
                        "phi_rad": float(popt[3]),
                        "a_nl": float(popt[4]),
                        "i0": float(popt[5]),
                        "q0": float(popt[6]),
                        "tau_s": float(popt[7]),
                        "fit_tau": bool(fit_tau),
                        "downward": bool(downward),
                        "nrmse": float(nrmse),
                        "success": True,
                    },
                )
            )
            self._mark_dirty()
            self._refresh_status()
            self._autosave_dataset()
            true_fr = fit_payload.get("true_fr_hz", np.nan)
            delta_fr = fit_payload.get("delta_fr_hz", np.nan)
            self._log(
                f"Fitted Logan nonlinear IQ model: fr0={popt[0] / _HZ_PER_GHZ:.9g} GHz, "
                f"powered_fr={float(true_fr) / _HZ_PER_GHZ:.9g} GHz, delta_fr={float(delta_fr):.4g} Hz, "
                f"Qr={popt[1]:.4g}, amp={popt[2]:.4g}, phi={np.degrees(popt[3]):.4g} deg, "
                f"a={popt[4]:.4g}, nrmse={float(nrmse):.4g}."
            )
            self._res_set_status(
                f"Fit complete: fr0={popt[0] / _HZ_PER_GHZ:.9g} GHz, "
                f"powered_fr={float(true_fr) / _HZ_PER_GHZ:.9g} GHz, delta_fr={float(delta_fr):.4g} Hz, "
                f"Qr={popt[1]:.4g}, a={popt[4]:.4g}, nrmse={float(nrmse):.4g}.",
                "dark green",
            )
            self._res_render()
        except Exception as exc:
            self._res_set_status(f"Fit failed: {exc}", "dark orange")
        finally:
            self._res_set_busy(False)

    def _res_set_status(self, message: str, color: str | None = None) -> None:
        if self.res_status_var is not None:
            self.res_status_var.set(message)
        if self.res_status_label is not None and color is not None:
            self.res_status_label.configure(fg=color)

    def _res_set_busy(self, busy: bool, message: str | None = None) -> None:
        if message is not None:
            self._res_set_status(message, "dark orange" if busy else "dark green")
        if self.res_window is not None and self.res_window.winfo_exists():
            self.res_window.update()

    def _res_attach_selection(self) -> None:
        scan = self._res_get_scan()
        if scan is None:
            return
        self._res_save_view_settings()
        scan.processing_history.append(
            _make_event(
                "attach_resonance_selection_view",
                {
                    "selection_range_hz": list(self._res_selected_range) if self._res_selected_range is not None else [],
                    "display_mode": self.res_display_mode_var.get() if self.res_display_mode_var is not None else "amplitude",
                    "auto_y": bool(self.res_auto_y_var.get()) if self.res_auto_y_var is not None else True,
                },
            )
        )
        self._mark_dirty()
        self._refresh_status()
        self._autosave_dataset()
        self._res_set_status("Attached current resonance selection as the default view.", "dark green")

    def _res_update_toolbar_history_buttons(self) -> None:
        if self.res_toolbar is None:
            return
        history = getattr(self, "_res_view_history", [])
        index = int(getattr(self, "_res_view_history_index", -1))
        buttons = getattr(self.res_toolbar, "_buttons", {})
        back_btn = buttons.get("Back")
        forward_btn = buttons.get("Forward")
        if back_btn is not None:
            back_btn.configure(state=(tk.NORMAL if index > 0 else tk.DISABLED))
        if forward_btn is not None:
            forward_btn.configure(state=(tk.NORMAL if index >= 0 and index < len(history) - 1 else tk.DISABLED))

    def _res_view_state(self) -> dict[str, object]:
        return {
            "xlim": tuple(self._res_selected_range) if self._res_selected_range is not None else None,
            "ylim": tuple(self._res_manual_ylim) if self._res_manual_ylim is not None else None,
        }

    def _res_push_view_history(self) -> None:
        if getattr(self, "_res_history_applying", False):
            return
        state = self._res_view_state()
        history = list(getattr(self, "_res_view_history", []))
        index = int(getattr(self, "_res_view_history_index", -1))
        if 0 <= index < len(history) and history[index] == state:
            return
        if index < len(history) - 1:
            history = history[: index + 1]
        history.append(state)
        self._res_view_history = history
        self._res_view_history_index = len(history) - 1
        self._res_update_toolbar_history_buttons()

    def _res_apply_view_state(self, state: dict[str, object]) -> None:
        self._res_history_applying = True
        try:
            xlim = state.get("xlim")
            ylim = state.get("ylim")
            self._res_selected_range = tuple(xlim) if isinstance(xlim, (tuple, list)) else None
            self._res_manual_ylim = tuple(ylim) if isinstance(ylim, (tuple, list)) else None
            self._res_render()
        finally:
            self._res_history_applying = False

    def _res_nav_back(self, *_args) -> None:
        history = getattr(self, "_res_view_history", [])
        index = int(getattr(self, "_res_view_history_index", -1))
        if index <= 0 or not history:
            return
        self._res_view_history_index = index - 1
        self._res_update_toolbar_history_buttons()
        self._res_apply_view_state(history[self._res_view_history_index])

    def _res_nav_forward(self, *_args) -> None:
        history = getattr(self, "_res_view_history", [])
        index = int(getattr(self, "_res_view_history_index", -1))
        if not history or index >= len(history) - 1:
            return
        self._res_view_history_index = index + 1
        self._res_update_toolbar_history_buttons()
        self._res_apply_view_state(history[self._res_view_history_index])

    def _res_nav_home(self, *_args) -> None:
        self._res_reset_view()

    def open_resonance_selection_window(self) -> None:
        scans = self._selected_scans()
        if not scans:
            messagebox.showwarning("No selection", "Select scans for analysis first.")
            return
        if fit_nonlinear_iq is None or nonlinear_iq is None or guess_p0_nonlinear_iq is None:
            messagebox.showerror(
                "citkid unavailable",
                f"Could not import citkid.res from:\n{_CITKID_MAIN}",
            )
            return

        chosen_scan = self._choose_resonance_scan(scans)
        if chosen_scan is None:
            return
        self._last_resonance_scan_key = self._scan_key(chosen_scan)

        if self.res_window is not None and self.res_window.winfo_exists():
            self._res_close()

        self.res_window = tk.Toplevel(self.root)
        self.res_window.title("Logan Resonance Fit")
        self.res_window.geometry("1250x780")
        self.res_window.protocol("WM_DELETE_WINDOW", self._res_close)

        top = tk.Frame(self.res_window, padx=8, pady=6)
        top.pack(side="top", fill="x")
        tk.Label(
            top,
            text=f"Scan: {Path(chosen_scan.filename).name} | Use toolbar zoom on left plot to select raw S21 fit region",
            anchor="w",
        ).pack(side="left", fill="x", expand=True)
        tk.Button(top, text="Choose Scan", command=self.open_resonance_selection_window).pack(side="right")
        self.res_fit_button = tk.Button(top, text="Fit Logan Model", command=self._res_fit_displayed_data)
        self.res_fit_button.pack(side="right", padx=(0, 8))
        tk.Button(top, text="Clear Params", command=self._res_clear_logan_params).pack(side="right", padx=(0, 8))
        tk.Button(top, text="Reset View", command=self._res_reset_view).pack(side="right", padx=(0, 8))
        self.res_auto_y_var = tk.BooleanVar(value=True)
        self.res_display_mode_var = None
        controls = tk.Frame(self.res_window, padx=8, pady=2)
        controls.pack(side="top", fill="x")
        tk.Checkbutton(
            controls,
            text="Auto-scale raw |S21| in window",
            variable=self.res_auto_y_var,
            command=self._res_on_controls_changed,
        ).pack(side="left", padx=(0, 12))
        self.res_fr_var = tk.StringVar()
        self.res_qi_var = tk.StringVar()
        self.res_qc_var = tk.StringVar()
        self.res_qc_phase_var = tk.StringVar()
        self.res_a_mag_var = tk.StringVar()
        self.res_a_phase_var = tk.StringVar()
        self.res_q0_var = tk.StringVar()
        self.res_tau_var = tk.StringVar()
        self.res_true_fr_var = tk.StringVar()
        self.res_delta_fr_var = tk.StringVar()
        self.res_nrmse_var = tk.StringVar()
        self.res_qi_output_var = tk.StringVar()
        self.res_qc_output_var = tk.StringVar()
        self.res_fix_a_mag_var = tk.BooleanVar(value=True)
        self.res_fix_tau_var = tk.BooleanVar(value=True)
        tk.Label(controls, text="fr0 (GHz)").pack(side="left", padx=(12, 2))
        tk.Entry(controls, textvariable=self.res_fr_var, width=10).pack(side="left")
        tk.Label(controls, text="Qr").pack(side="left", padx=(8, 2))
        tk.Entry(controls, textvariable=self.res_qi_var, width=8).pack(side="left")
        tk.Label(controls, text="amp").pack(side="left", padx=(8, 2))
        tk.Entry(controls, textvariable=self.res_qc_var, width=7).pack(side="left")
        tk.Label(controls, text="phi (deg)").pack(side="left", padx=(8, 2))
        tk.Entry(controls, textvariable=self.res_qc_phase_var, width=7).pack(side="left")
        tk.Label(controls, text="a").pack(side="left", padx=(8, 2))
        tk.Entry(controls, textvariable=self.res_a_mag_var, width=8).pack(side="left")
        tk.Label(controls, text="i0").pack(side="left", padx=(8, 2))
        self.res_a_phase_entry = tk.Entry(controls, textvariable=self.res_a_phase_var, width=8)
        self.res_a_phase_entry.pack(side="left")
        tk.Label(controls, text="q0").pack(side="left", padx=(8, 2))
        tk.Entry(controls, textvariable=self.res_q0_var, width=8).pack(side="left")
        tk.Label(controls, text="tau (s)").pack(side="left", padx=(8, 2))
        self.res_tau_entry = tk.Entry(controls, textvariable=self.res_tau_var, width=10)
        self.res_tau_entry.pack(side="left")
        tk.Checkbutton(controls, text="Fit tau", variable=self.res_fix_tau_var).pack(side="left", padx=(8, 0))
        tk.Checkbutton(controls, text="Downward", variable=self.res_fix_a_mag_var).pack(side="left", padx=(8, 0))
        tk.Label(controls, text="powered fr (GHz)").pack(side="left", padx=(12, 2))
        tk.Entry(controls, textvariable=self.res_true_fr_var, width=10, state="readonly").pack(side="left")
        tk.Label(controls, text="delta fr (Hz)").pack(side="left", padx=(8, 2))
        tk.Entry(controls, textvariable=self.res_delta_fr_var, width=12, state="readonly").pack(side="left")
        tk.Label(controls, text="nrmse").pack(side="left", padx=(8, 2))
        tk.Entry(controls, textvariable=self.res_nrmse_var, width=10, state="readonly").pack(side="left")
        tk.Label(controls, text="Qi").pack(side="left", padx=(8, 2))
        tk.Entry(controls, textvariable=self.res_qi_output_var, width=10, state="readonly").pack(side="left")
        tk.Label(controls, text="Qc").pack(side="left", padx=(8, 2))
        tk.Entry(controls, textvariable=self.res_qc_output_var, width=10, state="readonly").pack(side="left")

        self.res_status_var = tk.StringVar(
            value="Use toolbar zoom to select a raw S21 frequency span."
        )
        status_row = tk.Frame(self.res_window, padx=8, pady=4)
        status_row.pack(side="top", fill="x")
        self.res_status_label = tk.Label(status_row, textvariable=self.res_status_var, anchor="w")
        self.res_status_label.pack(side="left", fill="x", expand=True)

        self.res_figure = Figure(figsize=(12, 7))
        self.res_canvas = FigureCanvasTkAgg(self.res_figure, master=self.res_window)
        self.res_toolbar = NavigationToolbar2Tk(self.res_canvas, self.res_window)
        self.res_toolbar.set_history_buttons = self._res_update_toolbar_history_buttons
        buttons = getattr(self.res_toolbar, "_buttons", {})
        if buttons.get("Home") is not None:
            buttons["Home"].configure(command=self._res_nav_home)
        if buttons.get("Back") is not None:
            buttons["Back"].configure(command=self._res_nav_back)
        if buttons.get("Forward") is not None:
            buttons["Forward"].configure(command=self._res_nav_forward)
        self.res_toolbar.update()
        self.res_toolbar.pack(side="top", fill="x")
        self.res_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.res_canvas.mpl_connect("button_release_event", lambda _e: self._res_on_zoom_release())

        self._res_scan_key = self._scan_key(chosen_scan)
        view = self._res_get_view_settings(chosen_scan)
        self._res_selected_range = tuple(view["xlim"])
        self._res_manual_ylim = tuple(view["ylim"]) if view["ylim"] is not None else None
        self.res_auto_y_var.set(bool(view["auto_y"]))
        for field in (
            self.res_fr_var,
            self.res_qi_var,
            self.res_qc_var,
            self.res_qc_phase_var,
            self.res_a_mag_var,
            self.res_a_phase_var,
            self.res_q0_var,
            self.res_tau_var,
        ):
            if field is not None:
                field.set("")
        self._res_set_logan_output_fields(None)
        self.res_model_preview = None
        self._res_view_history = []
        self._res_view_history_index = -1
        self._res_history_applying = False
        self._res_push_view_history()
        self._res_update_toolbar_history_buttons()
        self._res_set_busy(True, "Opening raw S21 plot...")
        self.res_window.update_idletasks()
        self.res_window.after(10, self._res_render)

    def _choose_resonance_scan(self, scans) -> Optional[object]:
        options = []
        default_index = 0
        for i, scan in enumerate(scans):
            key = self._scan_key(scan)
            options.append(
                self._scan_dialog_label(
                    scan,
                    include_loaded_at=True,
                )
            )
            if key == self._last_resonance_scan_key:
                default_index = i
        pick = self._select_setting_option(
            "Choose Scan",
            "Select one VNA scan for resonance selection:",
            options,
            default_index=default_index,
        )
        if pick is None:
            return None
        return scans[pick]

    def _res_get_scan(self):
        for scan in self._selected_scans():
            if self._scan_key(scan) == self._res_scan_key:
                return scan
        return None

    def _res_get_view_settings(self, scan) -> dict:
        freq = np.asarray(scan.freq, dtype=float)
        default_xlim = (float(np.min(freq)), float(np.max(freq)))
        view = scan.candidate_resonators.get("resonance_selection_view", {})
        if not isinstance(view, dict):
            return {
                "xlim": default_xlim,
                "ylim": None,
                "auto_y": True,
                "use_corrected_data": True,
                "show_phase_left": False,
            }
        xlim = view.get("xlim", default_xlim)
        if not isinstance(xlim, (list, tuple)) or len(xlim) != 2:
            xlim = default_xlim
        ylim = view.get("ylim", None)
        if not isinstance(ylim, (list, tuple)) or len(ylim) != 2:
            ylim = None
        return {
            "xlim": (float(xlim[0]), float(xlim[1])),
            "ylim": (float(ylim[0]), float(ylim[1])) if ylim is not None else None,
            "auto_y": bool(view.get("auto_y", True)),
            "use_corrected_data": bool(view.get("use_corrected_data", True)),
            "show_phase_left": bool(view.get("show_phase_left", False)),
        }

    def _res_save_view_settings(self) -> None:
        scan = self._res_get_scan()
        if scan is None:
            return
        xlim = self._res_selected_range
        ylim = self._res_manual_ylim
        if self.res_amp_ax is not None:
            xlim = tuple(float(v) * _HZ_PER_GHZ for v in self.res_amp_ax.get_xlim())
            if self.res_auto_y_var is not None and not bool(self.res_auto_y_var.get()):
                ylim = tuple(self.res_amp_ax.get_ylim())
        scan.candidate_resonators["resonance_selection_view"] = {
            "xlim": xlim,
            "ylim": ylim,
            "auto_y": bool(self.res_auto_y_var.get()) if self.res_auto_y_var is not None else True,
            "use_corrected_data": True,
            "show_phase_left": (
                self.res_display_mode_var is not None and self.res_display_mode_var.get() == "phase"
            ),
        }

    def _res_get_normalized_complex(self, scan) -> np.ndarray:
        norm = scan.baseline_filter["normalized"]
        amp, phase = _read_polar_series(
            norm,
            amplitude_key="norm_amp",
            phase_key="norm_phase_deg_unwrapped",
        )
        if amp.shape != scan.freq.shape or phase.shape != scan.freq.shape:
            raise ValueError("Invalid normalized attachment: amplitude/phase shape mismatch.")
        return _complex_from_polar(amp, phase)

    def _res_get_normalized_amp(self, scan) -> np.ndarray:
        norm = scan.baseline_filter["normalized"]
        amp, _phase = _read_polar_series(
            norm,
            amplitude_key="norm_amp",
            phase_key="norm_phase_deg_unwrapped",
        )
        if amp.shape != scan.freq.shape:
            raise ValueError("Invalid normalized attachment: amplitude shape mismatch.")
        return amp

    def _res_get_normalized_phase(self, scan) -> np.ndarray:
        norm = scan.baseline_filter["normalized"]
        _amp, phase = _read_polar_series(
            norm,
            amplitude_key="norm_amp",
            phase_key="norm_phase_deg_unwrapped",
        )
        if phase.shape != scan.freq.shape:
            raise ValueError("Invalid normalized attachment: phase shape mismatch.")
        return phase

    def _res_autoscale_amp_y_for_visible_x(self, ax) -> None:
        if self.res_auto_y_var is None or not bool(self.res_auto_y_var.get()):
            return
        lines = [ln for ln in ax.get_lines() if ln.get_visible()]
        if not lines:
            return
        x0, x1 = ax.get_xlim()
        lo, hi = (x0, x1) if x0 <= x1 else (x1, x0)
        y_chunks = []
        for ln in lines:
            x = np.asarray(ln.get_xdata(), dtype=float)
            y = np.asarray(ln.get_ydata(), dtype=float)
            if x.size == 0 or y.size == 0 or x.size != y.size:
                continue
            mask = np.isfinite(x) & np.isfinite(y) & (x >= lo) & (x <= hi)
            if np.any(mask):
                y_chunks.append(y[mask])
        if not y_chunks:
            return
        y_all = np.concatenate(y_chunks)
        y_min = float(np.min(y_all))
        y_max = float(np.max(y_all))
        pad = 1.0 if y_max <= y_min else 0.05 * (y_max - y_min)
        ax.set_ylim(y_min - pad, y_max + pad)

    def _res_on_controls_changed(self) -> None:
        if self.res_amp_ax is not None and self.res_auto_y_var is not None and bool(self.res_auto_y_var.get()):
            self._res_autoscale_amp_y_for_visible_x(self.res_amp_ax)
        self._res_render()

    def _res_on_zoom_release(self) -> None:
        if self.res_amp_ax is None:
            return
        new_range = tuple(float(v) * _HZ_PER_GHZ for v in self.res_amp_ax.get_xlim())
        prev_range = tuple(self._res_selected_range) if self._res_selected_range is not None else None
        if prev_range is not None and np.allclose(new_range, prev_range, rtol=0.0, atol=1e-12):
            return
        self._res_selected_range = new_range
        if self.res_auto_y_var is not None and bool(self.res_auto_y_var.get()):
            self._res_autoscale_amp_y_for_visible_x(self.res_amp_ax)
        else:
            self._res_manual_ylim = tuple(self.res_amp_ax.get_ylim())
        self._res_push_view_history()
        self._res_render()

    def _res_reset_view(self) -> None:
        scan = self._res_get_scan()
        if scan is None:
            return
        freq = np.asarray(scan.freq, dtype=float)
        if freq.size == 0:
            return
        self._res_selected_range = (float(np.min(freq)), float(np.max(freq)))
        self._res_manual_ylim = None
        self._res_push_view_history()
        self._res_set_status("Reset to the full frequency range.", "dark green")
        self._res_render()

    def _res_extract_candidates(self, scan) -> tuple[np.ndarray, np.ndarray]:
        cand = scan.candidate_resonators
        g = cand.get("gaussian_convolution", {})
        d = cand.get("dsdf_gaussian_convolution", {})
        gfreq = np.asarray(g.get("candidate_freq", np.array([])), dtype=float)
        dfreq = np.asarray(d.get("candidate_freq", np.array([])), dtype=float)
        return gfreq, dfreq

    def _res_nearest_indices(self, query_freqs: np.ndarray, ref_freqs: np.ndarray) -> np.ndarray:
        q = np.asarray(query_freqs, dtype=float).ravel()
        ref = np.asarray(ref_freqs, dtype=float).ravel()
        if q.size == 0 or ref.size == 0:
            return np.empty((0,), dtype=int)
        idx = []
        for f in q:
            idx.append(int(np.argmin(np.abs(ref - f))))
        return np.asarray(idx, dtype=int)

    def _res_get_phase_class_points(self, scan) -> dict:
        points = scan.candidate_resonators["phase_class_points"]
        if not isinstance(points, dict):
            raise ValueError("phase_class_points must be a dict attached by Phase Correction.")
        return {
            "regular_freqs": np.asarray(points["regular_freqs"], dtype=float),
            "irregular_congruent_freqs": np.asarray(points["irregular_congruent_freqs"], dtype=float),
            "irregular_noncongruent_freqs": np.asarray(points["irregular_noncongruent_freqs"], dtype=float),
        }

    def _res_render(self) -> None:
        if self.res_figure is None or self.res_canvas is None:
            return
        self._res_set_busy(True, "Rendering raw Logan fit plot...")
        scan = self._res_get_scan()
        try:
            if scan is None:
                self.res_figure.clear()
                ax = self.res_figure.add_subplot(111)
                ax.text(0.5, 0.5, "Selected scan is unavailable.", ha="center", va="center")
                ax.axis("off")
                self.res_canvas.draw_idle()
                return

            freq = np.asarray(scan.freq, dtype=float)
            freq_ghz = freq / _HZ_PER_GHZ
            z = np.asarray(scan.s21_complex_raw, dtype=np.complex128)
            y_left = np.abs(z)

            self.res_figure.clear()
            ax_amp = self.res_figure.add_subplot(1, 2, 1)
            ax_iq = self.res_figure.add_subplot(1, 2, 2)
            self.res_amp_ax = ax_amp
            self.res_iq_ax = ax_iq

            ax_amp.set_xlabel("Frequency (GHz)")
            ax_amp.set_ylabel("|S21|")
            ax_amp.grid(True, alpha=0.3)
            ax_amp.set_title("Raw S21 Fit Window", fontsize=10)

            if self._res_selected_range is None:
                self._res_selected_range = (float(freq[0]), float(freq[-1]))

            (lo, hi), mask = self._res_get_selection_mask(freq)
            lo_ghz = lo / _HZ_PER_GHZ
            hi_ghz = hi / _HZ_PER_GHZ
            ax_amp.set_xlim(lo_ghz, hi_ghz)
            if self.res_auto_y_var is not None and bool(self.res_auto_y_var.get()):
                self._res_autoscale_amp_y_for_visible_x(ax_amp)
            elif self._res_manual_ylim is not None:
                ax_amp.set_ylim(self._res_manual_ylim)

            if np.count_nonzero(mask) >= 2:
                ax_amp.plot(
                    freq_ghz[mask],
                    y_left[mask],
                    color="tab:blue",
                    linewidth=1.2,
                    label="Displayed region",
                )

                ax_iq.plot(
                    np.real(z[mask]),
                    np.imag(z[mask]),
                    color="tab:blue",
                    linewidth=1.0,
                    label="Selected region",
                )
                ax_iq.scatter(
                    np.real(z[mask][0]),
                    np.imag(z[mask][0]),
                    c="tab:green",
                    s=16,
                    label="Start",
                    zorder=3,
                )
                ax_iq.scatter(
                    np.real(z[mask][-1]),
                    np.imag(z[mask][-1]),
                    c="tab:red",
                    s=16,
                    label="End",
                    zorder=3,
                )
                fit_payload = None
                if (
                    isinstance(self.res_model_preview, dict)
                    and self.res_model_preview.get("scan_key") == self._scan_key(scan)
                    and np.allclose(
                        np.asarray(self.res_model_preview.get("selection_range_hz", ()), dtype=float),
                        np.asarray((lo, hi), dtype=float),
                        rtol=0.0,
                        atol=1e-6,
                    )
                ):
                    fit_payload = self.res_model_preview
                else:
                    stored = scan.candidate_resonators.get("logan_nonlinear_iq_fit")
                    if isinstance(stored, dict):
                        fit_range = stored.get("selection_range_hz")
                        if isinstance(fit_range, (list, tuple)) and len(fit_range) == 2 and np.allclose(
                            np.asarray(fit_range, dtype=float),
                            np.asarray((lo, hi), dtype=float),
                            rtol=0.0,
                            atol=1e-6,
                        ):
                            fit_payload = stored
                if fit_payload is not None:
                    fit_freq = np.asarray(fit_payload["fit_freq_hz"], dtype=float)
                    fit_freq_ghz = fit_freq / _HZ_PER_GHZ
                    fit_z = np.asarray(fit_payload["fit_s21_complex"], dtype=np.complex128)
                    fit_y = np.abs(fit_z)
                    fr0_hz = float(fit_payload.get("fr0_hz", fit_payload.get("fr_hz", 0.5 * (lo + hi))))
                    fr0_ghz = fr0_hz / _HZ_PER_GHZ
                    true_fr_raw = fit_payload.get("true_fr_hz")
                    true_fr_hz = float(true_fr_raw) if true_fr_raw is not None else np.nan
                    true_fr_ghz = true_fr_hz / _HZ_PER_GHZ
                    ax_amp.plot(
                        fit_freq_ghz,
                        fit_y,
                        color="darkorange",
                        linewidth=1.4,
                        linestyle="--",
                        label="Logan fit",
                    )
                    if fit_freq.size:
                        fr_idx = int(np.argmin(np.abs(fit_freq - fr0_hz)))
                        fr0_iq = np.interp(fr0_hz, fit_freq, np.real(fit_z)) + 1j * np.interp(
                            fr0_hz, fit_freq, np.imag(fit_z)
                        )
                        if lo <= fr0_hz <= hi:
                            ax_amp.axvline(
                                fr0_ghz,
                                color="purple",
                                linestyle=":",
                                linewidth=1.4,
                                label="fr0",
                            )
                        if np.isfinite(true_fr_hz) and lo <= true_fr_hz <= hi:
                            ax_amp.axvline(
                                true_fr_ghz,
                                color="crimson",
                                linestyle="-.",
                                linewidth=1.4,
                                label="powered fr",
                            )
                    ax_iq.plot(
                        np.real(fit_z),
                        np.imag(fit_z),
                        color="darkorange",
                        linewidth=1.2,
                        linestyle="--",
                        label="Logan fit",
                    )
                    if fit_freq.size:
                        ax_iq.plot(
                            [np.real(fr0_iq)],
                            [np.imag(fr0_iq)],
                            linestyle="none",
                            marker="x",
                            markersize=10,
                            markeredgewidth=2.0,
                            color="purple",
                            label="fr0",
                        )
                        if np.isfinite(true_fr_hz) and lo <= true_fr_hz <= hi:
                            true_iq = np.interp(true_fr_hz, fit_freq, np.real(fit_z)) + 1j * np.interp(
                                true_fr_hz, fit_freq, np.imag(fit_z)
                            )
                            ax_iq.plot(
                                [np.real(true_iq)],
                                [np.imag(true_iq)],
                                linestyle="none",
                                marker="+",
                                markersize=12,
                                markeredgewidth=2.0,
                                color="crimson",
                                label="powered fr",
                            )
                self._res_set_status(
                    f"Displayed {np.count_nonzero(mask)} raw points: {lo_ghz:.9g} to {hi_ghz:.9g} GHz."
                    ,
                    "dark green",
                )
            else:
                ax_iq.text(0.5, 0.5, "Select a wider frequency region.", ha="center", va="center")
                self._res_set_status("Selection too small. Drag a wider region.", "dark orange")

            ax_amp.legend(loc="best", fontsize=8)
            ax_iq.set_xlabel("Re(raw S21)")
            ax_iq.set_ylabel("Im(raw S21)")
            ax_iq.grid(True, alpha=0.3)
            ax_iq.set_title("Complex Plane (Displayed Frequency Window)", fontsize=10)
            ax_iq.set_aspect("equal", adjustable="box")

            ax_iq.legend(loc="best", fontsize=8)

            self.res_figure.tight_layout()
            self.res_canvas.draw_idle()
        finally:
            self._res_set_busy(False)

    def _res_close(self) -> None:
        if self.res_window is not None and self.res_window.winfo_exists():
            self.res_window.destroy()
        self.res_window = None
        self.res_canvas = None
        self.res_toolbar = None
        self.res_figure = None
        self.res_fit_button = None
        self.res_status_var = None
        self.res_fr_var = None
        self.res_qi_var = None
        self.res_qc_var = None
        self.res_qc_phase_var = None
        self.res_a_mag_var = None
        self.res_a_phase_var = None
        self.res_q0_var = None
        self.res_tau_var = None
        self.res_true_fr_var = None
        self.res_delta_fr_var = None
        self.res_nrmse_var = None
        self.res_qi_output_var = None
        self.res_qc_output_var = None
        self.res_fix_fr_var = None
        self.res_fix_qi_var = None
        self.res_fix_qc_var = None
        self.res_fix_qc_phase_var = None
        self.res_fix_a_mag_var = None
        self.res_fix_a_phase_var = None
        self.res_fix_tau_var = None
        self.res_a_phase_entry = None
        self.res_tau_entry = None
        self.res_fix_a_phase_check = None
        self.res_fix_tau_check = None
        self.res_status_label = None
        self.res_auto_y_var = None
        self.res_display_mode_var = None
        self.res_fit_mode_var = None
        self.res_model_preview = None
        self.res_amp_ax = None
        self.res_iq_ax = None
        self._res_scan_key = None
        self._res_selected_range = None
        self._res_manual_ylim = None
        self._res_view_history = []
        self._res_view_history_index = -1
        self._res_history_applying = False
    def _res_get_raw_complex(self, scan) -> np.ndarray:
        return scan.complex_s21()

    def _res_get_raw_phase(self, scan) -> np.ndarray:
        return np.degrees(np.angle(self._res_get_raw_complex(scan)))
