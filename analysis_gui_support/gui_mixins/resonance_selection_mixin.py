from __future__ import annotations

from pathlib import Path
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
        freq = np.asarray(scan.freq, dtype=float)
        (lo, hi), mask = self._res_get_selection_mask(freq)
        if np.count_nonzero(mask) < 2:
            self._res_set_status("Need a wider displayed region to plot the model.", "dark orange")
            return
        try:
            self.res_model_preview = self._res_build_model_preview(scan, lo=lo, hi=hi)
            self._res_set_status("Displayed current resonator model parameters.", "dark green")
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
        freq = np.asarray(scan.freq, dtype=float)
        (lo, hi), mask = self._res_get_selection_mask(freq)
        if np.count_nonzero(mask) < 2:
            lo = float(freq[0])
            hi = float(freq[-1])
        gfreq, dfreq = self._res_extract_candidates(scan)
        fr0 = self._res_fit_initial_frequency(lo, hi, gfreq, dfreq)
        self._res_set_model_fields(
            fr_hz=fr0,
            qi=1.0e9,
            q_cpl_mag=1.0e4,
            q_cpl_phase_deg=0.0,
            a_mag=1.0,
            a_phase_deg=0.0,
            tau_s=0.0,
        )
        if self.res_fix_fr_var is not None:
            self.res_fix_fr_var.set(False)
        if self.res_fix_qi_var is not None:
            self.res_fix_qi_var.set(False)
        if self.res_fix_qc_var is not None:
            self.res_fix_qc_var.set(False)
        if self.res_fix_qc_phase_var is not None:
            self.res_fix_qc_phase_var.set(False)
        if self.res_fix_a_mag_var is not None:
            self.res_fix_a_mag_var.set(True)
        if self.res_fix_a_phase_var is not None:
            self.res_fix_a_phase_var.set(False)
        if self.res_fix_tau_var is not None:
            self.res_fix_tau_var.set(False)
        self.res_model_preview = None
        self._res_set_status("Reset resonator model parameters to defaults.", "dark green")
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
        freq = np.asarray(scan.freq, dtype=float)
        (lo, hi), mask = self._res_get_selection_mask(freq)
        if np.count_nonzero(mask) < 8:
            self._res_set_status("Need a wider displayed region before fitting.", "dark orange")
            return

        z = self._res_get_normalized_complex(scan)
        f_fit = freq[mask]
        z_fit = z[mask]
        fit_mode = self.res_fit_mode_var.get() if self.res_fit_mode_var is not None else "both"
        fr_guess, qi_guess, qcom_guess, a_guess, tau_guess = self._res_get_model_params_from_fields(lo=lo, hi=hi)
        fixed = self._res_fit_fix_flags()
        if fit_mode == "amplitude":
            a_guess = abs(a_guess) + 0j
            tau_guess = 0.0
            fixed["a_phase"] = True
            fixed["tau"] = True
        fr_center = 0.5 * (lo + hi)
        fr_half_span = max(0.5 * (hi - lo), 1.0)
        base_params = {
            "fr_rel": float((fr_guess - fr_center) / fr_half_span),
            "log_qi": float(np.log10(max(qi_guess, 1.0))),
            "log_qc": float(np.log10(max(abs(qcom_guess), 1.0))),
            "qc_phase": float(np.angle(qcom_guess)),
            "log_a_mag": float(np.log10(max(abs(a_guess), 1.0e-6))),
            "a_phase": float(np.angle(a_guess)),
            "tau": float(tau_guess),
        }

        free_keys = []
        for key in ("fr_rel", "log_qi", "log_qc", "qc_phase", "log_a_mag", "a_phase", "tau"):
            if key == "fr_rel" and fixed["fr"]:
                continue
            if key == "log_qi" and fixed["qi"]:
                continue
            if key == "log_qc" and fixed["qc"]:
                continue
            if key == "qc_phase" and fixed["qc_phase"]:
                continue
            if key == "log_a_mag" and fixed["a_mag"]:
                continue
            if key == "a_phase" and fixed["a_phase"]:
                continue
            if key == "tau" and fixed["tau"]:
                continue
            free_keys.append(key)

        def unpack(params: np.ndarray) -> tuple[float, float, complex, complex, float]:
            values = dict(base_params)
            for key, value in zip(free_keys, params):
                values[key] = float(value)
            fr = float(fr_center + values["fr_rel"] * fr_half_span)
            qi = 10.0 ** float(values["log_qi"])
            q_cpl_mag = 10.0 ** float(values["log_qc"])
            q_cpl_phase = float(values["qc_phase"])
            a_mag = 10.0 ** float(values["log_a_mag"])
            a_phase = float(values["a_phase"])
            tau = float(values["tau"])
            qcom = q_cpl_mag * np.exp(1j * q_cpl_phase)
            a = a_mag * np.exp(1j * a_phase)
            return fr, qi, qcom, a, tau

        def residuals(params: np.ndarray) -> np.ndarray:
            fr, qi, qcom, a, tau = unpack(params)
            model = ComplexResonanceQi(f_fit, fr, qi, qcom, a, tau)
            if fit_mode == "amplitude":
                return np.abs(model) - np.abs(z_fit)
            diff = model - z_fit
            return np.concatenate((np.real(diff), np.imag(diff)))

        lower_map = {
            "fr_rel": -1.0,
            "log_qi": 2.0,
            "log_qc": 2.0,
            "qc_phase": -np.pi,
            "log_a_mag": -3.0,
            "a_phase": -np.pi,
            "tau": -1.0e-5,
        }
        upper_map = {
            "fr_rel": 1.0,
            "log_qi": 12.0,
            "log_qc": 9.0,
            "qc_phase": np.pi,
            "log_a_mag": 1.0,
            "a_phase": np.pi,
            "tau": 1.0e-5,
        }
        p0 = np.array([base_params[key] for key in free_keys], dtype=float)
        lower = np.array([lower_map[key] for key in free_keys], dtype=float)
        upper = np.array([upper_map[key] for key in free_keys], dtype=float)
        diff_step_map = {
            "fr_rel": 1.0e-3,
            "log_qi": 1.0e-2,
            "log_qc": 1.0e-2,
            "qc_phase": 1.0e-2,
            "log_a_mag": 1.0e-2,
            "a_phase": 1.0e-2,
            "tau": 1.0e-10,
        }
        diff_step = np.array([diff_step_map[key] for key in free_keys], dtype=float)

        self._res_set_busy(True, "Fitting resonator model...")
        try:
            result = (
                least_squares(
                    residuals,
                    p0,
                    bounds=(lower, upper),
                    max_nfev=800,
                    x_scale="jac",
                    diff_step=diff_step,
                )
                if free_keys
                else None
            )
            params_out = result.x if result is not None else np.asarray([], dtype=float)
            fr, qi, qcom, a, tau = unpack(params_out)
            q_loaded = 1.0 / (1.0 / qi + np.real(1.0 / qcom))
            model = ComplexResonanceQi(f_fit, fr, qi, qcom, a, tau)
            fit_payload = {
                "selection_range_hz": (float(lo), float(hi)),
                "fr_hz": float(fr),
                "q_internal": float(qi),
                "q_loaded": float(q_loaded),
                "q_coupling_complex": complex(qcom),
                "q_coupling_mag": float(np.abs(qcom)),
                "q_coupling_phase_deg": float(np.degrees(np.angle(qcom))),
                "a_complex": complex(a),
                "a_mag": float(np.abs(a)),
                "a_phase_deg": float(np.degrees(np.angle(a))),
                "tau_s": float(tau),
                "cost": float(result.cost) if result is not None else 0.0,
                "success": bool(result.success) if result is not None else True,
                "message": str(result.message) if result is not None else "All selected fit parameters were fixed.",
                "fit_freq_hz": np.asarray(f_fit, dtype=float),
                "fit_s21_complex": np.asarray(model, dtype=np.complex128),
            }
            scan.candidate_resonators["resonator_model_fit"] = fit_payload
            self.res_model_preview = fit_payload
            self._res_set_model_fields(
                fr_hz=float(fr),
                qi=float(qi),
                q_cpl_mag=float(np.abs(qcom)),
                q_cpl_phase_deg=float(np.degrees(np.angle(qcom))),
                a_mag=float(np.abs(a)),
                a_phase_deg=float(np.degrees(np.angle(a))),
                tau_s=float(tau),
            )
            scan.processing_history.append(
                _make_event(
                    "fit_resonator_model",
                    {
                        "selection_range_hz": [float(lo), float(hi)],
                        "fr_hz": float(fr),
                        "q_internal": float(qi),
                        "q_loaded": float(q_loaded),
                        "q_coupling_mag": float(np.abs(qcom)),
                        "q_coupling_phase_deg": float(np.degrees(np.angle(qcom))),
                        "a_mag": float(np.abs(a)),
                        "a_phase_deg": float(np.degrees(np.angle(a))),
                        "tau_s": float(tau),
                        "fit_mode": fit_mode,
                        "success": bool(result.success) if result is not None else True,
                    },
                )
            )
            self._mark_dirty()
            self._refresh_status()
            self._autosave_dataset()
            fit_status = "converged" if (result is None or result.success) else "did not converge"
            fit_message = str(result.message).strip() if result is not None else "All selected fit parameters were fixed."
            self._log(
                f"Fitted resonator model ({fit_status}, mode={fit_mode}): fr={fr / _HZ_PER_GHZ:.9g} GHz, "
                f"Qi={qi:.4g}, Q={q_loaded:.4g}, |Qc|={abs(qcom):.4g}, |a|={abs(a):.4g}. "
                f"Solver message: {fit_message}"
            )
            self._res_set_status(
                f"Fit {'complete' if (result is None or result.success) else 'ended without convergence'}: "
                f"fr={fr / _HZ_PER_GHZ:.9g} GHz, Qi={qi:.4g}, Q={q_loaded:.4g}, |Qc|={abs(qcom):.4g}, |a|={abs(a):.4g}.",
                "dark green" if (result is None or result.success) else "dark orange",
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
        if not self._selected_scans_have_attached_normalized():
            messagebox.showwarning(
                "Missing normalized data",
                "Run pipeline in order:\n"
                "Phase Correction -> Baseline Filtering -> Interp+Smooth -> Normalize Baseline -> Resonator Selection.\n\n"
                "All selected scans must have attached normalized data first.",
            )
            return

        chosen_scan = self._choose_resonance_scan(scans)
        if chosen_scan is None:
            return
        if "phase_class_points" not in chosen_scan.candidate_resonators:
            messagebox.showwarning(
                "Missing phase class points",
                "Run 'Phase Correction' and click Attach for this scan before resonance selection.",
            )
            return
        self._last_resonance_scan_key = self._scan_key(chosen_scan)

        if self.res_window is not None and self.res_window.winfo_exists():
            self._res_close()

        self.res_window = tk.Toplevel(self.root)
        self.res_window.title("Resonance Selection")
        self.res_window.geometry("1250x780")
        self.res_window.protocol("WM_DELETE_WINDOW", self._res_close)

        top = tk.Frame(self.res_window, padx=8, pady=6)
        top.pack(side="top", fill="x")
        tk.Label(
            top,
            text=f"Scan: {Path(chosen_scan.filename).name} | Use toolbar zoom on left plot to select frequency region",
            anchor="w",
        ).pack(side="left", fill="x", expand=True)
        tk.Button(top, text="Choose Scan", command=self.open_resonance_selection_window).pack(side="right")
        tk.Button(top, text="Attach Selection", command=self._res_attach_selection).pack(side="right", padx=(0, 8))
        self.res_fit_button = tk.Button(top, text="Fit Resonator Model", command=self._res_fit_displayed_data)
        self.res_fit_button.pack(side="right", padx=(0, 8))
        tk.Button(top, text="Reset Model Params", command=self._res_reset_model_parameters).pack(
            side="right", padx=(0, 8)
        )
        tk.Button(top, text="Plot Current Model", command=self._res_display_current_model).pack(
            side="right", padx=(0, 8)
        )
        tk.Button(top, text="Reset View", command=self._res_reset_view).pack(side="right", padx=(0, 8))
        self.res_auto_y_var = tk.BooleanVar(value=True)
        self.res_display_mode_var = tk.StringVar(value="amplitude")
        self.res_fit_mode_var = tk.StringVar(value="amplitude")
        controls = tk.Frame(self.res_window, padx=8, pady=2)
        controls.pack(side="top", fill="x")
        tk.Checkbutton(
            controls,
            text="Auto-scale |S21| in window",
            variable=self.res_auto_y_var,
            command=self._res_on_controls_changed,
        ).pack(side="left", padx=(0, 12))
        tk.Radiobutton(
            controls,
            text="Amplitude",
            variable=self.res_display_mode_var,
            value="amplitude",
            command=self._res_on_controls_changed,
        ).pack(side="left")
        tk.Radiobutton(
            controls,
            text="Phase",
            variable=self.res_display_mode_var,
            value="phase",
            command=self._res_on_controls_changed,
        ).pack(side="left", padx=(8, 0))
        tk.Label(controls, text="Fit").pack(side="left", padx=(12, 2))
        tk.Radiobutton(
            controls,
            text="Both amp+phase",
            variable=self.res_fit_mode_var,
            value="both",
            command=self._res_update_fit_mode_controls,
        ).pack(side="left")
        tk.Radiobutton(
            controls,
            text="Amplitude only",
            variable=self.res_fit_mode_var,
            value="amplitude",
            command=self._res_update_fit_mode_controls,
        ).pack(side="left", padx=(8, 0))
        self.res_fr_var = tk.StringVar()
        self.res_qi_var = tk.StringVar()
        self.res_qc_var = tk.StringVar()
        self.res_qc_phase_var = tk.StringVar()
        self.res_a_mag_var = tk.StringVar()
        self.res_a_phase_var = tk.StringVar()
        self.res_tau_var = tk.StringVar()
        self.res_fix_fr_var = tk.BooleanVar(value=False)
        self.res_fix_qi_var = tk.BooleanVar(value=False)
        self.res_fix_qc_var = tk.BooleanVar(value=False)
        self.res_fix_qc_phase_var = tk.BooleanVar(value=False)
        self.res_fix_a_mag_var = tk.BooleanVar(value=True)
        self.res_fix_a_phase_var = tk.BooleanVar(value=False)
        self.res_fix_tau_var = tk.BooleanVar(value=False)
        tk.Label(controls, text="fr (GHz)").pack(side="left", padx=(12, 2))
        tk.Entry(controls, textvariable=self.res_fr_var, width=10).pack(side="left")
        tk.Checkbutton(controls, text="Fix", variable=self.res_fix_fr_var).pack(side="left", padx=(2, 0))
        tk.Label(controls, text="Qi").pack(side="left", padx=(8, 2))
        tk.Entry(controls, textvariable=self.res_qi_var, width=10).pack(side="left")
        tk.Checkbutton(controls, text="Fix", variable=self.res_fix_qi_var).pack(side="left", padx=(2, 0))
        tk.Label(controls, text="|Qc|").pack(side="left", padx=(8, 2))
        tk.Entry(controls, textvariable=self.res_qc_var, width=10).pack(side="left")
        tk.Checkbutton(controls, text="Fix", variable=self.res_fix_qc_var).pack(side="left", padx=(2, 0))
        tk.Label(controls, text="Qc phase (deg)").pack(side="left", padx=(8, 2))
        tk.Entry(controls, textvariable=self.res_qc_phase_var, width=10).pack(side="left")
        tk.Checkbutton(controls, text="Fix", variable=self.res_fix_qc_phase_var).pack(side="left", padx=(2, 0))
        tk.Label(controls, text="|a|").pack(side="left", padx=(8, 2))
        tk.Entry(controls, textvariable=self.res_a_mag_var, width=8).pack(side="left")
        tk.Checkbutton(controls, text="Fix", variable=self.res_fix_a_mag_var).pack(side="left", padx=(2, 0))
        tk.Label(controls, text="a phase (deg)").pack(side="left", padx=(8, 2))
        self.res_a_phase_entry = tk.Entry(controls, textvariable=self.res_a_phase_var, width=8)
        self.res_a_phase_entry.pack(side="left")
        self.res_fix_a_phase_check = tk.Checkbutton(controls, text="Fix", variable=self.res_fix_a_phase_var)
        self.res_fix_a_phase_check.pack(side="left", padx=(2, 0))
        tk.Label(controls, text="tau (s)").pack(side="left", padx=(8, 2))
        self.res_tau_entry = tk.Entry(controls, textvariable=self.res_tau_var, width=10)
        self.res_tau_entry.pack(side="left")
        self.res_fix_tau_check = tk.Checkbutton(controls, text="Fix", variable=self.res_fix_tau_var)
        self.res_fix_tau_check.pack(side="left", padx=(2, 0))

        self.res_status_var = tk.StringVar(
            value="Use toolbar zoom to select a frequency span."
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
        self.res_display_mode_var.set("phase" if bool(view["show_phase_left"]) else "amplitude")
        gfreq, dfreq = self._res_extract_candidates(chosen_scan)
        fr0 = self._res_fit_initial_frequency(self._res_selected_range[0], self._res_selected_range[1], gfreq, dfreq)
        self._res_set_model_fields(
            fr_hz=fr0,
            qi=1.0e9,
            q_cpl_mag=1.0e4,
            q_cpl_phase_deg=0.0,
            a_mag=1.0,
            a_phase_deg=0.0,
            tau_s=0.0,
        )
        self._res_update_fit_mode_controls()
        self._res_view_history = []
        self._res_view_history_index = -1
        self._res_history_applying = False
        self._res_push_view_history()
        self._res_update_toolbar_history_buttons()
        self._res_set_busy(True, "Opening resonance selection plot...")
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
        self._res_set_busy(True, "Rendering resonance selection plot...")
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
            use_corrected = True
            show_phase = (
                self.res_display_mode_var is not None and self.res_display_mode_var.get() == "phase"
            )
            y_left = self._res_get_normalized_phase(scan) if show_phase else self._res_get_normalized_amp(scan)
            z = self._res_get_normalized_complex(scan)
            gfreq, dfreq = self._res_extract_candidates(scan)
            phase_points = self._res_get_phase_class_points(scan)

            self.res_figure.clear()
            ax_amp = self.res_figure.add_subplot(1, 2, 1)
            ax_iq = self.res_figure.add_subplot(1, 2, 2)
            self.res_amp_ax = ax_amp
            self.res_iq_ax = ax_iq

            left_label = "Normalized phase (deg)" if show_phase else None
            if show_phase:
                ax_amp.plot(freq_ghz, y_left, color="0.65", linewidth=0.8, label=left_label)
            ax_amp.set_xlabel("Frequency (GHz)")
            ax_amp.set_ylabel("Phase (deg)" if show_phase else "|S21|")
            ax_amp.grid(True, alpha=0.3)
            ax_amp.set_title("Zoom Here To Select/Display Frequency Window", fontsize=10)

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

            if gfreq.size:
                gmask = (gfreq >= lo) & (gfreq <= hi)
                if np.any(gmask):
                    gfreq_in = gfreq[gmask]
                    gi = np.clip(np.searchsorted(freq, gfreq_in), 0, freq.size - 1)
                    ax_amp.plot(
                        freq_ghz[gi],
                        y_left[gi],
                        linestyle="none",
                        marker="o",
                        markersize=8,
                        markerfacecolor="none",
                        markeredgewidth=1.5,
                        color="green",
                        label="Gaussian candidates",
                    )
            if dfreq.size:
                dmask = (dfreq >= lo) & (dfreq <= hi)
                if np.any(dmask):
                    dfreq_in = dfreq[dmask]
                    di = np.clip(np.searchsorted(freq, dfreq_in), 0, freq.size - 1)
                    ax_amp.plot(
                        freq_ghz[di],
                        y_left[di],
                        linestyle="none",
                        marker="D",
                        markersize=6,
                        color="red",
                        label="dS21/df peaks",
                    )

            if np.count_nonzero(mask) >= 2:
                ax_amp.plot(
                    freq_ghz[mask],
                    y_left[mask],
                    color="tab:blue",
                    linewidth=1.2,
                    label="Displayed region",
                )

                reg_freqs = phase_points["regular_freqs"]
                if reg_freqs.size:
                    rmask = (reg_freqs >= lo) & (reg_freqs <= hi)
                    if np.any(rmask):
                        rf = reg_freqs[rmask]
                        ri = self._res_nearest_indices(rf, freq)
                        ax_amp.plot(
                            freq_ghz[ri],
                            y_left[ri],
                            linestyle="none",
                            marker="o",
                            markersize=4,
                            color="black",
                            label="2*pi phase wrap corrections",
                        )

                cong_freqs = phase_points["irregular_congruent_freqs"]
                if cong_freqs.size:
                    cmask = (cong_freqs >= lo) & (cong_freqs <= hi)
                    if np.any(cmask):
                        cf = cong_freqs[cmask]
                        ci = self._res_nearest_indices(cf, freq)
                        ax_amp.plot(
                            freq_ghz[ci],
                            y_left[ci],
                            linestyle="none",
                            marker="o",
                            markersize=5,
                            color="pink",
                            label="VNA phase corrections",
                        )

                nonc_freqs = phase_points["irregular_noncongruent_freqs"]
                if nonc_freqs.size:
                    nmask = (nonc_freqs >= lo) & (nonc_freqs <= hi)
                    if np.any(nmask):
                        nf = nonc_freqs[nmask]
                        ni = self._res_nearest_indices(nf, freq)
                        ax_amp.plot(
                            freq_ghz[ni],
                            y_left[ni],
                            linestyle="none",
                            marker="o",
                            markersize=5,
                            color="blue",
                            label="Other phase discontinuities",
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
                if gfreq.size:
                    gmask = (gfreq >= lo) & (gfreq <= hi)
                    if np.any(gmask):
                        gfreq_in = gfreq[gmask]
                        gi = np.clip(np.searchsorted(freq, gfreq_in), 0, freq.size - 1)
                        ax_iq.plot(
                            np.real(z[gi]),
                            np.imag(z[gi]),
                            linestyle="none",
                            marker="o",
                            markersize=8,
                            markerfacecolor="none",
                            markeredgewidth=1.5,
                            color="green",
                            label="Gaussian candidates",
                        )
                if dfreq.size:
                    dmask = (dfreq >= lo) & (dfreq <= hi)
                    if np.any(dmask):
                        dfreq_in = dfreq[dmask]
                        di = np.clip(np.searchsorted(freq, dfreq_in), 0, freq.size - 1)
                        ax_iq.plot(
                            np.real(z[di]),
                            np.imag(z[di]),
                            linestyle="none",
                            marker="D",
                            markersize=6,
                            color="red",
                            label="dS21/df peaks",
                        )
                if reg_freqs.size:
                    rmask = (reg_freqs >= lo) & (reg_freqs <= hi)
                    if np.any(rmask):
                        ri = self._res_nearest_indices(reg_freqs[rmask], freq)
                        ax_iq.plot(
                            np.real(z[ri]),
                            np.imag(z[ri]),
                            linestyle="none",
                            marker="o",
                            markersize=4,
                            color="black",
                            label="2*pi phase wrap corrections",
                        )
                if cong_freqs.size:
                    cmask = (cong_freqs >= lo) & (cong_freqs <= hi)
                    if np.any(cmask):
                        ci = self._res_nearest_indices(cong_freqs[cmask], freq)
                        ax_iq.plot(
                            np.real(z[ci]),
                            np.imag(z[ci]),
                            linestyle="none",
                            marker="o",
                            markersize=5,
                            color="pink",
                            label="VNA phase corrections",
                        )
                if nonc_freqs.size:
                    nmask = (nonc_freqs >= lo) & (nonc_freqs <= hi)
                    if np.any(nmask):
                        ni = self._res_nearest_indices(nonc_freqs[nmask], freq)
                        ax_iq.plot(
                            np.real(z[ni]),
                            np.imag(z[ni]),
                            linestyle="none",
                            marker="o",
                            markersize=5,
                            color="blue",
                            label="Other phase discontinuities",
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
                elif self._res_current_fit(scan) is not None:
                    fit_payload = self._res_current_fit(scan)
                if fit_payload is not None:
                    fit_freq = np.asarray(fit_payload["fit_freq_hz"], dtype=float)
                    fit_freq_ghz = fit_freq / _HZ_PER_GHZ
                    fit_z = np.asarray(fit_payload["fit_s21_complex"], dtype=np.complex128)
                    fit_y = np.degrees(np.unwrap(np.angle(fit_z))) if show_phase else np.abs(fit_z)
                    fr_model_hz = float(fit_payload.get("fr_hz", 0.5 * (lo + hi)))
                    fr_model_ghz = fr_model_hz / _HZ_PER_GHZ
                    ax_amp.plot(
                        fit_freq_ghz,
                        fit_y,
                        color="darkorange",
                        linewidth=1.4,
                        linestyle="--",
                    )
                    if fit_freq.size:
                        fr_idx = int(np.argmin(np.abs(fit_freq - fr_model_hz)))
                        ax_amp.plot(
                            [fr_model_ghz],
                            [fit_y[fr_idx]],
                            linestyle="none",
                            marker="x",
                            markersize=10,
                            markeredgewidth=2.0,
                            color="purple",
                        )
                    ax_iq.plot(
                        np.real(fit_z),
                        np.imag(fit_z),
                        color="darkorange",
                        linewidth=1.2,
                        linestyle="--",
                        label="Resonator fit",
                    )
                    if fit_freq.size:
                        ax_iq.plot(
                            [np.real(fit_z[fr_idx])],
                            [np.imag(fit_z[fr_idx])],
                            linestyle="none",
                            marker="x",
                            markersize=10,
                            markeredgewidth=2.0,
                            color="purple",
                            label="Model fr",
                        )
                self._res_set_status(
                    f"Displayed {np.count_nonzero(mask)} points: {lo_ghz:.9g} to {hi_ghz:.9g} GHz."
                    ,
                    "dark green",
                )
            else:
                ax_iq.text(0.5, 0.5, "Select a wider frequency region.", ha="center", va="center")
                self._res_set_status("Selection too small. Drag a wider region.", "dark orange")

            if self.res_display_mode_var is not None and self.res_display_mode_var.get() == "amplitude":
                ax_amp.legend(loc="best", fontsize=8)
            iq_label = "normalized" if use_corrected else "raw"
            ax_iq.set_xlabel(f"Re({iq_label} S21)")
            ax_iq.set_ylabel(f"Im({iq_label} S21)")
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
        self.res_tau_var = None
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
