from __future__ import annotations

from pathlib import Path

import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from analysis_gui_support.analysis_models import _read_polar_series

class ScanEvolutionMixin:
    def _scan_evolution_make_stage(
        self,
        *,
        name: str,
        freq_hz: np.ndarray,
        amp: np.ndarray,
        phase_deg: np.ndarray,
    ) -> Optional[dict]:
        freq = np.asarray(freq_hz, dtype=float)
        amp_arr = np.asarray(amp, dtype=float)
        phase_arr = np.asarray(phase_deg, dtype=float)
        if freq.ndim != 1 or amp_arr.ndim != 1 or phase_arr.ndim != 1:
            return None
        if freq.size == 0 or amp_arr.shape != freq.shape or phase_arr.shape != freq.shape:
            return None
        order = np.argsort(freq)
        freq = freq[order]
        amp_arr = amp_arr[order]
        phase_arr = phase_arr[order]
        real = amp_arr * np.cos(np.radians(phase_arr))
        imag = amp_arr * np.sin(np.radians(phase_arr))
        return {
            "name": name,
            "freq_hz": freq,
            "freq_ghz": freq / 1.0e9,
            "amp": amp_arr,
            "phase_deg": phase_arr,
            "real": real,
            "imag": imag,
        }


    def _scan_evolution_phase_display(self, stage: dict) -> np.ndarray:
        phase = np.asarray(stage["phase_deg"], dtype=float)
        if self.scan_evolution_mod360_var is not None and self.scan_evolution_mod360_var.get():
            return ((phase + 180.0) % 360.0) - 180.0
        return phase


    @staticmethod
    def _scan_evolution_nearest_values(
        query_freq_hz: Sequence[float],
        ref_freq_hz: Sequence[float],
        ref_values: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        query = np.asarray(query_freq_hz, dtype=float).ravel()
        ref_f = np.asarray(ref_freq_hz, dtype=float).ravel()
        ref_v = np.asarray(ref_values, dtype=float).ravel()
        if query.size == 0 or ref_f.size == 0 or ref_v.size == 0 or ref_f.size != ref_v.size:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        idx = np.searchsorted(ref_f, query)
        idx = np.clip(idx, 0, max(ref_f.size - 1, 0))
        left = np.clip(idx - 1, 0, max(ref_f.size - 1, 0))
        use_left = np.abs(query - ref_f[left]) <= np.abs(query - ref_f[idx])
        chosen = np.where(use_left, left, idx)
        return ref_f[chosen], ref_v[chosen]


    def _scan_evolution_overlay_points(self, scan: VNAScan) -> dict[str, np.ndarray]:
        cand = scan.candidate_resonators
        gaussian = cand.get("gaussian_convolution", {})
        dsdf = cand.get("dsdf_gaussian_convolution", {})
        phase_points = cand.get("phase_class_points", {})
        return {
            "gaussian": np.asarray(gaussian.get("candidate_freq", np.array([])), dtype=float),
            "dsdf": np.asarray(dsdf.get("candidate_freq", np.array([])), dtype=float),
            "regular": np.asarray(phase_points.get("regular_freqs", np.array([])), dtype=float),
            "congruent": np.asarray(phase_points.get("irregular_congruent_freqs", np.array([])), dtype=float),
            "noncongruent": np.asarray(phase_points.get("irregular_noncongruent_freqs", np.array([])), dtype=float),
        }


    def _scan_evolution_attached_resonator_points(
        self,
        scan: VNAScan,
        *,
        phase_values: np.ndarray,
        amp_values: np.ndarray,
        real_values: np.ndarray,
        imag_values: np.ndarray,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        attached = scan.candidate_resonators.get("sheet_resonances", {})
        assignments = attached.get("assignments") if isinstance(attached, dict) else {}
        if not isinstance(assignments, dict):
            return [], [], []
        ref_freq_hz = np.asarray(scan.freq, dtype=float)
        amp_ref = np.asarray(amp_values, dtype=float)
        phase_ref = np.asarray(phase_values, dtype=float)
        real_ref = np.asarray(real_values, dtype=float)
        imag_ref = np.asarray(imag_values, dtype=float)
        if (
            ref_freq_hz.size == 0
            or amp_ref.shape != ref_freq_hz.shape
            or phase_ref.shape != ref_freq_hz.shape
            or real_ref.shape != ref_freq_hz.shape
            or imag_ref.shape != ref_freq_hz.shape
        ):
            return [], [], []
        amp_points: list[dict] = []
        phase_points: list[dict] = []
        complex_points: list[dict] = []
        for resonator_number, record in assignments.items():
            if not isinstance(record, dict):
                continue
            try:
                target_hz = float(record.get("frequency_hz"))
            except Exception:
                continue
            x_hz_amp, y_amp = self._scan_evolution_nearest_values([target_hz], ref_freq_hz, amp_ref)
            x_hz_phase, y_phase = self._scan_evolution_nearest_values([target_hz], ref_freq_hz, phase_ref)
            x_hz_real, y_real = self._scan_evolution_nearest_values([target_hz], ref_freq_hz, real_ref)
            x_hz_imag, y_imag = self._scan_evolution_nearest_values([target_hz], ref_freq_hz, imag_ref)
            if x_hz_amp.size:
                amp_points.append({"x_hz": float(x_hz_amp[0]), "y": float(y_amp[0]), "label": str(resonator_number)})
            if x_hz_phase.size:
                phase_points.append({"x_hz": float(x_hz_phase[0]), "y": float(y_phase[0]), "label": str(resonator_number)})
            if x_hz_real.size and x_hz_imag.size:
                complex_points.append(
                    {
                        "x_hz": float(x_hz_real[0]),
                        "real": float(y_real[0]),
                        "imag": float(y_imag[0]),
                        "label": str(resonator_number),
                    }
                )
        return amp_points, phase_points, complex_points


    def _scan_evolution_add_overlays(
        self,
        ax,
        scan: VNAScan,
        *,
        values: np.ndarray,
        use_phase: bool,
        used_labels: set[str],
    ) -> None:
        ref_freq_hz = np.asarray(scan.freq, dtype=float)
        ref_y = np.asarray(values, dtype=float)
        marker_defs = [
            (
                self.scan_evolution_show_gaussian_var is not None and bool(self.scan_evolution_show_gaussian_var.get()),
                "gaussian",
                dict(linestyle="none", marker="o", markersize=8, markerfacecolor="none", markeredgewidth=1.5, color="green"),
                "Gaussian candidates",
            ),
            (
                self.scan_evolution_show_dsdf_var is not None and bool(self.scan_evolution_show_dsdf_var.get()),
                "dsdf",
                dict(linestyle="none", marker="D", markersize=6, color="red"),
                "dS21/df peaks",
            ),
            (
                self.scan_evolution_show_phase_2pi_var is not None and bool(self.scan_evolution_show_phase_2pi_var.get()),
                "regular",
                dict(linestyle="none", marker="o", markersize=4, color="black"),
                "2pi phase corrections",
            ),
            (
                self.scan_evolution_show_phase_vna_var is not None and bool(self.scan_evolution_show_phase_vna_var.get()),
                "congruent",
                dict(linestyle="none", marker="o", markersize=5, color="pink"),
                "VNA phase corrections",
            ),
            (
                self.scan_evolution_show_phase_other_var is not None and bool(self.scan_evolution_show_phase_other_var.get()),
                "noncongruent",
                dict(linestyle="none", marker="o", markersize=2.5, color="blue"),
                "Other phase discontinuities",
            ),
        ]
        points = self._scan_evolution_overlay_points(scan)
        for enabled, key, style, label in marker_defs:
            if not enabled:
                continue
            freq_pts = points[key]
            if freq_pts.size == 0:
                continue
            x_hz, y_pts = self._scan_evolution_nearest_values(freq_pts, ref_freq_hz, ref_y)
            if x_hz.size == 0:
                continue
            plot_label = label if label not in used_labels else None
            ax.plot(x_hz / 1.0e9, y_pts, label=plot_label, **style)
            if plot_label is not None:
                used_labels.add(label)


    def _scan_evolution_complex_overlay_points(
        self,
        scan: VNAScan,
        *,
        real_values: np.ndarray,
        imag_values: np.ndarray,
        freq_lo_ghz: float,
        freq_hi_ghz: float,
    ) -> list[tuple[np.ndarray, np.ndarray, dict, str]]:
        ref_freq_hz = np.asarray(scan.freq, dtype=float)
        real_ref = np.asarray(real_values, dtype=float)
        imag_ref = np.asarray(imag_values, dtype=float)
        if ref_freq_hz.size == 0 or real_ref.shape != ref_freq_hz.shape or imag_ref.shape != ref_freq_hz.shape:
            return []

        marker_defs = [
            (
                self.scan_evolution_show_gaussian_var is not None and bool(self.scan_evolution_show_gaussian_var.get()),
                "gaussian",
                dict(linestyle="none", marker="o", markersize=8, markerfacecolor="none", markeredgewidth=1.5, color="green"),
                "Gaussian candidates",
            ),
            (
                self.scan_evolution_show_dsdf_var is not None and bool(self.scan_evolution_show_dsdf_var.get()),
                "dsdf",
                dict(linestyle="none", marker="D", markersize=6, color="red"),
                "dS21/df peaks",
            ),
            (
                self.scan_evolution_show_phase_2pi_var is not None and bool(self.scan_evolution_show_phase_2pi_var.get()),
                "regular",
                dict(linestyle="none", marker="o", markersize=4, color="black"),
                "2pi phase corrections",
            ),
            (
                self.scan_evolution_show_phase_vna_var is not None and bool(self.scan_evolution_show_phase_vna_var.get()),
                "congruent",
                dict(linestyle="none", marker="o", markersize=5, color="pink"),
                "VNA phase corrections",
            ),
            (
                self.scan_evolution_show_phase_other_var is not None and bool(self.scan_evolution_show_phase_other_var.get()),
                "noncongruent",
                dict(linestyle="none", marker="o", markersize=2.5, color="blue"),
                "Other phase discontinuities",
            ),
        ]
        points = self._scan_evolution_overlay_points(scan)
        plotted: list[tuple[np.ndarray, np.ndarray, dict, str]] = []
        for enabled, key, style, label in marker_defs:
            if not enabled:
                continue
            freq_pts = np.asarray(points.get(key, np.array([])), dtype=float)
            if freq_pts.size == 0:
                continue
            freq_pts_ghz = freq_pts / 1.0e9
            visible_mask = np.isfinite(freq_pts_ghz) & (freq_pts_ghz >= freq_lo_ghz) & (freq_pts_ghz <= freq_hi_ghz)
            freq_visible = freq_pts[visible_mask]
            if freq_visible.size == 0:
                continue
            _x_hz, real_pts = self._scan_evolution_nearest_values(freq_visible, ref_freq_hz, real_ref)
            _x_hz_im, imag_pts = self._scan_evolution_nearest_values(freq_visible, ref_freq_hz, imag_ref)
            if real_pts.size == 0 or imag_pts.size == 0:
                continue
            plotted.append((real_pts, imag_pts, style, label))
        return plotted


    def _scan_evolution_toggle_phase_wrap(self) -> None:
        if self.scan_evolution_figure is None or self.scan_evolution_canvas is None:
            return
        self._render_scan_evolution_window()


    def _scan_evolution_stage_rows_for_scan(self, scan: VNAScan) -> list[dict]:
        stages: list[dict] = []
        raw_stage = self._scan_evolution_make_stage(
            name="Raw",
            freq_hz=scan.freq,
            amp=scan.amplitude(),
            phase_deg=scan.phase_deg_wrapped_raw(),
        )
        if raw_stage is not None:
            stages.append(raw_stage)

        if scan.has_dewrapped_phase():
            stage = self._scan_evolution_make_stage(
                name="Phase Corr. 1",
                freq_hz=scan.freq,
                amp=scan.amplitude(),
                phase_deg=scan.phase_deg_unwrapped(),
            )
            if stage is not None:
                stages.append(stage)

        phase2 = scan.candidate_resonators.get("phase_correction_2", {})
        amp2, phase2_deg = _read_polar_series(
            phase2,
            amplitude_key="corrected_amp",
            phase_key="corrected_phase_deg",
        )
        stage = self._scan_evolution_make_stage(
            name="Phase Corr. 2",
            freq_hz=scan.freq,
            amp=amp2,
            phase_deg=phase2_deg,
        )
        if stage is not None:
            stages.append(stage)

        phase3 = scan.candidate_resonators.get("phase_correction_3", {})
        amp3, phase3_deg = _read_polar_series(
            phase3,
            amplitude_key="corrected_amp",
            phase_key="corrected_phase_deg",
        )
        stage = self._scan_evolution_make_stage(
            name="Phase Corr. 3",
            freq_hz=scan.freq,
            amp=amp3,
            phase_deg=phase3_deg,
        )
        if stage is not None:
            stages.append(stage)

        bf = scan.baseline_filter if isinstance(scan.baseline_filter, dict) else {}
        norm = bf.get("normalized", {}) if isinstance(bf, dict) else {}
        norm_amp, norm_phase = _read_polar_series(
            norm,
            amplitude_key="norm_amp",
            phase_key="norm_phase_deg_unwrapped",
        )
        stage = self._scan_evolution_make_stage(
            name="Normalized",
            freq_hz=scan.freq,
            amp=norm_amp,
            phase_deg=norm_phase,
        )
        if stage is not None:
            stages.append(stage)

        return stages


    def open_scan_evolution_window(self) -> None:
        scan = self._choose_one_selected_scan()
        if scan is None:
            return

        if self.scan_evolution_window is not None and self.scan_evolution_window.winfo_exists():
            self.scan_evolution_window.lift()
            self._scan_evolution_scan_key = self._scan_key(scan)
            self._render_scan_evolution_window()
            return

        self.scan_evolution_window = tk.Toplevel(self.root)
        self.scan_evolution_window.title("Scan Evolution")
        self.scan_evolution_window.geometry("1500x980")
        self.scan_evolution_window.protocol("WM_DELETE_WINDOW", self._close_scan_evolution_window)
        self._scan_evolution_scan_key = self._scan_key(scan)

        controls = tk.Frame(self.scan_evolution_window, padx=8, pady=8)
        controls.pack(side="top", fill="x")
        self.scan_evolution_status_var = tk.StringVar(value="Showing selected scan evolution.")
        self.scan_evolution_mod360_var = tk.BooleanVar(value=True)
        self.scan_evolution_show_gaussian_var = tk.BooleanVar(value=False)
        self.scan_evolution_show_dsdf_var = tk.BooleanVar(value=False)
        self.scan_evolution_show_phase_2pi_var = tk.BooleanVar(value=False)
        self.scan_evolution_show_phase_vna_var = tk.BooleanVar(value=False)
        self.scan_evolution_show_phase_other_var = tk.BooleanVar(value=False)
        self.scan_evolution_show_attached_res_var = tk.BooleanVar(value=False)
        tk.Label(controls, textvariable=self.scan_evolution_status_var, anchor="w").pack(
            side="left", fill="x", expand=True
        )
        marker_controls = tk.Frame(self.scan_evolution_window, padx=8, pady=0)
        marker_controls.pack(side="top", fill="x")
        tk.Checkbutton(
            controls,
            text="Mod 360 phase",
            variable=self.scan_evolution_mod360_var,
            command=self._scan_evolution_toggle_phase_wrap,
        ).pack(side="right", padx=(0, 8))
        tk.Button(controls, text="Choose Scan", width=12, command=self._scan_evolution_choose_scan).pack(
            side="right"
        )
        tk.Button(controls, text="Reset View", width=12, command=self._scan_evolution_reset_view).pack(
            side="right", padx=(0, 8)
        )
        for text, var in (
            ("Gaussian", self.scan_evolution_show_gaussian_var),
            ("dS21/df", self.scan_evolution_show_dsdf_var),
            ("2pi", self.scan_evolution_show_phase_2pi_var),
            ("VNA phase", self.scan_evolution_show_phase_vna_var),
            ("Other phase", self.scan_evolution_show_phase_other_var),
            ("Resonators", self.scan_evolution_show_attached_res_var),
        ):
            tk.Checkbutton(
                marker_controls,
                text=text,
                variable=var,
                command=self._render_scan_evolution_window,
            ).pack(side="left", padx=(0, 8))

        self.scan_evolution_figure = Figure(figsize=(14, 9))
        self.scan_evolution_canvas = FigureCanvasTkAgg(self.scan_evolution_figure, master=self.scan_evolution_window)
        self.scan_evolution_toolbar = NavigationToolbar2Tk(self.scan_evolution_canvas, self.scan_evolution_window)
        self.scan_evolution_toolbar.update()
        self.scan_evolution_toolbar.pack(side="top", fill="x")
        self.scan_evolution_canvas.get_tk_widget().pack(fill="both", expand=True)

        self._render_scan_evolution_window()


    def _scan_evolution_choose_scan(self) -> None:
        scan = self._choose_one_selected_scan()
        if scan is None:
            return
        self._scan_evolution_scan_key = self._scan_key(scan)
        self._render_scan_evolution_window()


    def _close_scan_evolution_window(self) -> None:
        if self.scan_evolution_window is not None and self.scan_evolution_window.winfo_exists():
            self.scan_evolution_window.destroy()
        self.scan_evolution_window = None
        self.scan_evolution_canvas = None
        self.scan_evolution_toolbar = None
        self.scan_evolution_figure = None
        self.scan_evolution_status_var = None
        self.scan_evolution_mod360_var = None
        self.scan_evolution_show_gaussian_var = None
        self.scan_evolution_show_dsdf_var = None
        self.scan_evolution_show_phase_2pi_var = None
        self.scan_evolution_show_phase_vna_var = None
        self.scan_evolution_show_phase_other_var = None
        self.scan_evolution_show_attached_res_var = None
        self._scan_evolution_scan_key = None
        self._scan_evolution_stage_rows = []
        self._scan_evolution_axes_rows = []
        self._scan_evolution_syncing_xlim = False


    def _scan_evolution_current_scan(self) -> Optional[VNAScan]:
        if self._scan_evolution_scan_key is None:
            return None
        for scan in self.dataset.vna_scans:
            if self._scan_key(scan) == self._scan_evolution_scan_key:
                return scan
        return None


    def _scan_evolution_visible_xlim(self) -> Optional[tuple[float, float]]:
        if not self._scan_evolution_axes_rows:
            return None
        ax_amp = self._scan_evolution_axes_rows[0][0]
        try:
            x0, x1 = ax_amp.get_xlim()
        except Exception:
            return None
        return (float(x0), float(x1)) if float(x0) <= float(x1) else (float(x1), float(x0))


    def _scan_evolution_autoscale_amp_phase(self) -> None:
        xlim = self._scan_evolution_visible_xlim()
        if xlim is None:
            return
        lo, hi = xlim
        for stage, (ax_amp, ax_phase, _ax_complex) in zip(self._scan_evolution_stage_rows, self._scan_evolution_axes_rows):
            mask = np.isfinite(stage["freq_ghz"]) & (stage["freq_ghz"] >= lo) & (stage["freq_ghz"] <= hi)
            if np.any(mask):
                amp = np.asarray(stage["amp"], dtype=float)[mask]
                phase = self._scan_evolution_phase_display(stage)[mask]
                if amp.size:
                    amp_min = float(np.min(amp))
                    amp_max = float(np.max(amp))
                    amp_pad = 1.0 if amp_max <= amp_min else 0.05 * (amp_max - amp_min)
                    ax_amp.set_ylim(amp_min - amp_pad, amp_max + amp_pad)
                if phase.size:
                    if self.scan_evolution_mod360_var is not None and self.scan_evolution_mod360_var.get():
                        ax_phase.set_ylim(-180.0, 180.0)
                    else:
                        ph_min = float(np.min(phase))
                        ph_max = float(np.max(phase))
                        ph_pad = 1.0 if ph_max <= ph_min else 0.05 * (ph_max - ph_min)
                        ax_phase.set_ylim(ph_min - ph_pad, ph_max + ph_pad)


    def _scan_evolution_update_complex_axes(self) -> None:
        xlim = self._scan_evolution_visible_xlim()
        if xlim is None:
            return
        lo, hi = xlim
        scan = self._scan_evolution_current_scan()
        for stage, (_ax_amp, _ax_phase, ax_complex) in zip(self._scan_evolution_stage_rows, self._scan_evolution_axes_rows):
            ax_complex.clear()
            mask = np.isfinite(stage["freq_ghz"]) & (stage["freq_ghz"] >= lo) & (stage["freq_ghz"] <= hi)
            if np.any(mask):
                real = np.asarray(stage["real"], dtype=float)[mask]
                imag = np.asarray(stage["imag"], dtype=float)[mask]
                ax_complex.plot(real, imag, color="tab:green", linewidth=1.0)
                if scan is not None:
                    for real_pts, imag_pts, style, _label in self._scan_evolution_complex_overlay_points(
                        scan,
                        real_values=np.asarray(stage["real"], dtype=float),
                        imag_values=np.asarray(stage["imag"], dtype=float),
                        freq_lo_ghz=lo,
                        freq_hi_ghz=hi,
                    ):
                        ax_complex.plot(real_pts, imag_pts, **style)
                if self.scan_evolution_show_attached_res_var is not None and bool(self.scan_evolution_show_attached_res_var.get()):
                    if scan is not None:
                        _amp_points, _phase_points, complex_points = self._scan_evolution_attached_resonator_points(
                            scan,
                            phase_values=self._scan_evolution_phase_display(stage),
                            amp_values=np.asarray(stage["amp"], dtype=float),
                            real_values=np.asarray(stage["real"], dtype=float),
                            imag_values=np.asarray(stage["imag"], dtype=float),
                        )
                        complex_points = [
                            pt
                            for pt in complex_points
                            if lo <= float(pt["x_hz"]) / 1.0e9 <= hi
                        ]
                        if complex_points:
                            complex_marker_line = ax_complex.plot(
                                [float(pt["real"]) for pt in complex_points],
                                [float(pt["imag"]) for pt in complex_points],
                                linestyle="none",
                                marker="s",
                                markersize=5,
                                color="black",
                                clip_on=True,
                            )[0]
                            if hasattr(complex_marker_line, "set_in_layout"):
                                complex_marker_line.set_in_layout(False)
                            for pt in complex_points:
                                ann = ax_complex.annotate(
                                    str(pt["label"]),
                                    (float(pt["real"]), float(pt["imag"])),
                                    xytext=(4, 3),
                                    textcoords="offset points",
                                    fontsize=8,
                                    color="black",
                                    clip_on=True,
                                )
                                if hasattr(ann, "set_in_layout"):
                                    ann.set_in_layout(False)
                ax_complex.set_aspect("equal", adjustable="box")
                re_min = float(np.min(real))
                re_max = float(np.max(real))
                im_min = float(np.min(imag))
                im_max = float(np.max(imag))
                re_pad = 1.0 if re_max <= re_min else 0.05 * (re_max - re_min)
                im_pad = 1.0 if im_max <= im_min else 0.05 * (im_max - im_min)
                ax_complex.set_xlim(re_min - re_pad, re_max + re_pad)
                ax_complex.set_ylim(im_min - im_pad, im_max + im_pad)
            else:
                ax_complex.text(0.5, 0.5, "No data in range", ha="center", va="center", transform=ax_complex.transAxes)
            ax_complex.grid(True, alpha=0.3)
            ax_complex.set_xlabel("Real(S21)")
            ax_complex.set_ylabel("Imag(S21)")


    def _scan_evolution_on_xlim_changed(self, changed_ax) -> None:
        if self._scan_evolution_syncing_xlim:
            return
        try:
            xlim = changed_ax.get_xlim()
        except Exception:
            return
        self._scan_evolution_syncing_xlim = True
        try:
            for ax_amp, ax_phase, _ax_complex in self._scan_evolution_axes_rows:
                if ax_amp is not changed_ax:
                    ax_amp.set_xlim(xlim)
                if ax_phase is not changed_ax:
                    ax_phase.set_xlim(xlim)
            self._scan_evolution_autoscale_amp_phase()
            self._scan_evolution_update_complex_axes()
        finally:
            self._scan_evolution_syncing_xlim = False
        if self.scan_evolution_canvas is not None:
            self.scan_evolution_canvas.draw_idle()


    def _scan_evolution_reset_view(self) -> None:
        if not self._scan_evolution_axes_rows or not self._scan_evolution_stage_rows:
            return
        freq_min = min(float(stage["freq_ghz"][0]) for stage in self._scan_evolution_stage_rows)
        freq_max = max(float(stage["freq_ghz"][-1]) for stage in self._scan_evolution_stage_rows)
        x_pad = max((freq_max - freq_min) * 0.01, 1e-6)
        xlim = (freq_min - x_pad, freq_max + x_pad)
        self._scan_evolution_syncing_xlim = True
        try:
            for ax_amp, ax_phase, _ax_complex in self._scan_evolution_axes_rows:
                ax_amp.set_xlim(xlim)
                ax_phase.set_xlim(xlim)
        finally:
            self._scan_evolution_syncing_xlim = False
        self._scan_evolution_autoscale_amp_phase()
        self._scan_evolution_update_complex_axes()
        if self.scan_evolution_canvas is not None:
            self.scan_evolution_canvas.draw_idle()


    def _render_scan_evolution_window(self) -> None:
        if self.scan_evolution_figure is None or self.scan_evolution_canvas is None:
            return
        scan = self._scan_evolution_current_scan()
        if scan is None:
            return
        prior_limits: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = []
        for ax_amp, ax_phase, ax_complex in self._scan_evolution_axes_rows:
            try:
                prior_limits.append(
                    (
                        tuple(ax_amp.get_xlim()),
                        tuple(ax_amp.get_ylim()),
                        tuple(ax_phase.get_ylim()),
                    )
                )
            except Exception:
                prior_limits = []
                break
        stages = self._scan_evolution_stage_rows_for_scan(scan)
        self._scan_evolution_stage_rows = stages
        self._scan_evolution_axes_rows = []
        self.scan_evolution_figure.clear()
        if not stages:
            ax = self.scan_evolution_figure.add_subplot(111)
            ax.text(0.5, 0.5, "No attached processing stages available for this scan.", ha="center", va="center")
            ax.axis("off")
            self.scan_evolution_canvas.draw_idle()
            return

        nrows = len(stages)
        axes = np.atleast_2d(self.scan_evolution_figure.subplots(nrows, 3, sharex=False, squeeze=False))
        used_overlay_labels: set[str] = set()
        used_res_labels: set[str] = set()
        for row_idx, stage in enumerate(stages):
            ax_amp = axes[row_idx, 0]
            ax_phase = axes[row_idx, 1]
            ax_complex = axes[row_idx, 2]
            phase_display = self._scan_evolution_phase_display(stage)
            ax_amp.plot(stage["freq_ghz"], stage["amp"], color="tab:blue", linewidth=1.0)
            ax_phase.plot(stage["freq_ghz"], phase_display, color="tab:orange", linewidth=1.0)
            self._scan_evolution_add_overlays(
                ax_amp,
                scan,
                values=np.asarray(stage["amp"], dtype=float),
                use_phase=False,
                used_labels=used_overlay_labels,
            )
            self._scan_evolution_add_overlays(
                ax_phase,
                scan,
                values=phase_display,
                use_phase=True,
                used_labels=used_overlay_labels,
            )
            if self.scan_evolution_show_attached_res_var is not None and bool(self.scan_evolution_show_attached_res_var.get()):
                amp_points, phase_points, _complex_points = self._scan_evolution_attached_resonator_points(
                    scan,
                    phase_values=phase_display,
                    amp_values=np.asarray(stage["amp"], dtype=float),
                    real_values=np.asarray(stage["real"], dtype=float),
                    imag_values=np.asarray(stage["imag"], dtype=float),
                )
                if amp_points:
                    amp_marker_line = ax_amp.plot(
                        [pt["x_hz"] / 1.0e9 for pt in amp_points],
                        [pt["y"] for pt in amp_points],
                        linestyle="none",
                        marker="s",
                        markersize=5,
                        color="black",
                        label=("Attached resonators" if "Attached resonators" not in used_res_labels else None),
                        clip_on=True,
                    )[0]
                    if hasattr(amp_marker_line, "set_in_layout"):
                        amp_marker_line.set_in_layout(False)
                    used_res_labels.add("Attached resonators")
                if phase_points:
                    phase_marker_line = ax_phase.plot(
                        [pt["x_hz"] / 1.0e9 for pt in phase_points],
                        [pt["y"] for pt in phase_points],
                        linestyle="none",
                        marker="s",
                        markersize=5,
                        color="black",
                        label=("Attached resonators" if "Attached resonators" not in used_res_labels else None),
                        clip_on=True,
                    )[0]
                    if hasattr(phase_marker_line, "set_in_layout"):
                        phase_marker_line.set_in_layout(False)
                    used_res_labels.add("Attached resonators")
            ax_amp.grid(True, alpha=0.3)
            ax_phase.grid(True, alpha=0.3)
            ax_amp.set_ylabel(f"{stage['name']}\nAmplitude")
            ax_phase.set_ylabel("Phase (deg)")
            if row_idx == 0 and (used_overlay_labels or used_res_labels):
                ax_phase.legend(loc="best", fontsize=8)
            if row_idx == 0:
                ax_amp.set_title("Amplitude", fontsize=11)
                ax_phase.set_title("Phase", fontsize=11)
                ax_complex.set_title("Complex Plane", fontsize=11)
            if row_idx == nrows - 1:
                ax_amp.set_xlabel("Frequency (GHz)")
                ax_phase.set_xlabel("Frequency (GHz)")
            self._scan_evolution_axes_rows.append((ax_amp, ax_phase, ax_complex))

        self.scan_evolution_figure.suptitle(
            f"Scan Evolution | {Path(scan.filename).name}",
            fontsize=12,
        )
        self.scan_evolution_figure.tight_layout()

        for ax_amp, ax_phase, _ax_complex in self._scan_evolution_axes_rows:
            ax_amp.callbacks.connect("xlim_changed", self._scan_evolution_on_xlim_changed)
            ax_phase.callbacks.connect("xlim_changed", self._scan_evolution_on_xlim_changed)

        if len(prior_limits) == len(self._scan_evolution_axes_rows):
            self._scan_evolution_syncing_xlim = True
            try:
                for (ax_amp, ax_phase, _ax_complex), (xlim, amp_ylim, phase_ylim) in zip(
                    self._scan_evolution_axes_rows,
                    prior_limits,
                ):
                    ax_amp.set_xlim(xlim)
                    ax_phase.set_xlim(xlim)
                    ax_amp.set_ylim(amp_ylim)
                    ax_phase.set_ylim(phase_ylim)
            finally:
                self._scan_evolution_syncing_xlim = False
            self._scan_evolution_update_complex_axes()
            self.scan_evolution_canvas.draw_idle()
        else:
            self._scan_evolution_reset_view()
        if self.scan_evolution_status_var is not None:
            self.scan_evolution_status_var.set(
                f"Showing {len(stages)} stage(s) for {Path(scan.filename).name}. Zoom amplitude or phase to update all panels."
            )
