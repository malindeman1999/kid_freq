from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from tkinter import messagebox


class ResonanceSelectionMixin:
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
            text=f"Scan: {Path(chosen_scan.filename).name} | Drag on left plot to select frequency region",
            anchor="w",
        ).pack(side="left", fill="x", expand=True)
        tk.Button(top, text="Choose Scan", command=self.open_resonance_selection_window).pack(side="right")
        tk.Button(top, text="Reset View", command=self._res_reset_view).pack(side="right", padx=(0, 8))
        self.res_auto_y_var = tk.BooleanVar(value=True)
        self.res_use_corrected_var = tk.BooleanVar(value=True)
        self.res_show_phase_var = tk.BooleanVar(value=False)
        controls = tk.Frame(self.res_window, padx=8, pady=2)
        controls.pack(side="top", fill="x")
        tk.Checkbutton(
            controls,
            text="Auto-scale |S21| in window",
            variable=self.res_auto_y_var,
            command=self._res_on_controls_changed,
        ).pack(side="left", padx=(0, 12))
        tk.Checkbutton(
            controls,
            text="Use corrected data (normalized)",
            variable=self.res_use_corrected_var,
            command=self._res_on_controls_changed,
            state="disabled",
        ).pack(side="left")
        tk.Checkbutton(
            controls,
            text="Show Phase (left plot)",
            variable=self.res_show_phase_var,
            command=self._res_on_controls_changed,
        ).pack(side="left", padx=(12, 0))

        self.res_status_var = tk.StringVar(
            value="Use toolbar to zoom if needed, then drag on left plot to select a frequency span."
        )
        status_row = tk.Frame(self.res_window, padx=8, pady=4)
        status_row.pack(side="top", fill="x")
        tk.Label(status_row, textvariable=self.res_status_var, anchor="w").pack(
            side="left", fill="x", expand=True
        )

        self.res_figure = Figure(figsize=(12, 7))
        self.res_canvas = FigureCanvasTkAgg(self.res_figure, master=self.res_window)
        self.res_toolbar = NavigationToolbar2Tk(self.res_canvas, self.res_window)
        self.res_toolbar.update()
        self.res_toolbar.pack(side="top", fill="x")
        self.res_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.res_canvas.mpl_connect("button_release_event", lambda _e: self._res_on_zoom_release())

        self._res_scan_key = self._scan_key(chosen_scan)
        view = self._res_get_view_settings(chosen_scan)
        self._res_selected_range = tuple(view["xlim"])
        self._res_manual_ylim = tuple(view["ylim"]) if view["ylim"] is not None else None
        self.res_auto_y_var.set(bool(view["auto_y"]))
        self.res_use_corrected_var.set(True)
        self.res_show_phase_var.set(bool(view["show_phase_left"]))
        self._res_render()

    def _choose_resonance_scan(self, scans) -> Optional[object]:
        options = []
        default_index = 0
        for i, scan in enumerate(scans):
            key = self._scan_key(scan)
            options.append(f"{i+1}. {Path(scan.filename).name} | loaded {scan.loaded_at}")
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
            xlim = tuple(self.res_amp_ax.get_xlim())
            if self.res_auto_y_var is not None and not bool(self.res_auto_y_var.get()):
                ylim = tuple(self.res_amp_ax.get_ylim())
        scan.candidate_resonators["resonance_selection_view"] = {
            "xlim": xlim,
            "ylim": ylim,
            "auto_y": bool(self.res_auto_y_var.get()) if self.res_auto_y_var is not None else True,
            "use_corrected_data": True,
            "show_phase_left": bool(self.res_show_phase_var.get())
            if self.res_show_phase_var is not None
            else False,
        }

    def _res_get_normalized_complex(self, scan) -> np.ndarray:
        norm = scan.baseline_filter["normalized"]
        arr = np.asarray(norm["norm_complex"], dtype=np.complex128)
        if arr.shape != scan.freq.shape:
            raise ValueError("Invalid normalized attachment: norm_complex shape mismatch.")
        return arr

    def _res_get_normalized_amp(self, scan) -> np.ndarray:
        return np.abs(self._res_get_normalized_complex(scan))

    def _res_get_normalized_phase(self, scan) -> np.ndarray:
        return np.degrees(np.unwrap(np.angle(self._res_get_normalized_complex(scan))))

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
        self._res_save_view_settings()
        self._res_render()

    def _res_on_zoom_release(self) -> None:
        if self.res_amp_ax is None:
            return
        self._res_selected_range = tuple(self.res_amp_ax.get_xlim())
        if self.res_auto_y_var is not None and bool(self.res_auto_y_var.get()):
            self._res_autoscale_amp_y_for_visible_x(self.res_amp_ax)
        else:
            self._res_manual_ylim = tuple(self.res_amp_ax.get_ylim())
        self._res_save_view_settings()
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
        self._res_save_view_settings()
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
        scan = self._res_get_scan()
        if scan is None:
            self.res_figure.clear()
            ax = self.res_figure.add_subplot(111)
            ax.text(0.5, 0.5, "Selected scan is unavailable.", ha="center", va="center")
            ax.axis("off")
            self.res_canvas.draw_idle()
            return

        freq = scan.freq
        use_corrected = True
        show_phase = self.res_show_phase_var is not None and bool(self.res_show_phase_var.get())
        y_left = self._res_get_normalized_phase(scan) if show_phase else self._res_get_normalized_amp(scan)
        z = self._res_get_normalized_complex(scan)
        gfreq, dfreq = self._res_extract_candidates(scan)
        phase_points = self._res_get_phase_class_points(scan)

        self.res_figure.clear()
        ax_amp = self.res_figure.add_subplot(1, 2, 1)
        ax_iq = self.res_figure.add_subplot(1, 2, 2)
        self.res_amp_ax = ax_amp
        self.res_iq_ax = ax_iq

        left_label = (
            ("Normalized phase (deg)" if show_phase else "Normalized amplitude")
            if use_corrected
            else ("Raw phase (deg)" if show_phase else "Raw amplitude")
        )
        ax_amp.plot(freq, y_left, color="0.65", linewidth=0.8, label=left_label)
        ax_amp.set_xlabel("Frequency")
        ax_amp.set_ylabel("Phase (deg)" if show_phase else "|S21|")
        ax_amp.grid(True, alpha=0.3)
        ax_amp.set_title("Drag Here To Select/Display Frequency Window", fontsize=10)

        if self._res_selected_range is None:
            self._res_selected_range = (float(freq[0]), float(freq[-1]))

        fmin, fmax = self._res_selected_range
        lo, hi = (fmin, fmax) if fmin <= fmax else (fmax, fmin)
        ax_amp.set_xlim(lo, hi)
        mask = (freq >= lo) & (freq <= hi)
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
                    freq[gi],
                    y_left[gi],
                    linestyle="none",
                    marker="o",
                    markersize=5,
                    color="magenta",
                    label="Gaussian candidates",
                )
        if dfreq.size:
            dmask = (dfreq >= lo) & (dfreq <= hi)
            if np.any(dmask):
                dfreq_in = dfreq[dmask]
                di = np.clip(np.searchsorted(freq, dfreq_in), 0, freq.size - 1)
                ax_amp.plot(
                    freq[di],
                    y_left[di],
                    linestyle="none",
                    marker="x",
                    markersize=6,
                    color="cyan",
                    label="dS21/df peaks",
                )

        if np.count_nonzero(mask) >= 2:
            ax_amp.plot(freq[mask], y_left[mask], color="tab:blue", linewidth=1.2, label="Displayed region")

            reg_freqs = phase_points["regular_freqs"]
            if reg_freqs.size:
                rmask = (reg_freqs >= lo) & (reg_freqs <= hi)
                if np.any(rmask):
                    rf = reg_freqs[rmask]
                    ri = self._res_nearest_indices(rf, freq)
                    ax_amp.plot(
                        freq[ri],
                        y_left[ri],
                        linestyle="none",
                        marker="o",
                        markersize=4,
                        color="black",
                        label="Regular (360*n)",
                    )

            cong_freqs = phase_points["irregular_congruent_freqs"]
            if cong_freqs.size:
                cmask = (cong_freqs >= lo) & (cong_freqs <= hi)
                if np.any(cmask):
                    cf = cong_freqs[cmask]
                    ci = self._res_nearest_indices(cf, freq)
                    ax_amp.plot(
                        freq[ci],
                        y_left[ci],
                        linestyle="none",
                        marker="o",
                        markersize=5,
                        color="pink",
                        label="Irregular congruent",
                    )

            nonc_freqs = phase_points["irregular_noncongruent_freqs"]
            if nonc_freqs.size:
                nmask = (nonc_freqs >= lo) & (nonc_freqs <= hi)
                if np.any(nmask):
                    nf = nonc_freqs[nmask]
                    ni = self._res_nearest_indices(nf, freq)
                    ax_amp.plot(
                        freq[ni],
                        y_left[ni],
                        linestyle="none",
                        marker="o",
                        markersize=5,
                        color="blue",
                        label="Irregular non-congruent",
                    )

            ax_iq.plot(
                np.real(z[mask]),
                np.imag(z[mask]),
                color="tab:orange",
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
                        markersize=5,
                        color="magenta",
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
                        marker="x",
                        markersize=6,
                        color="cyan",
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
                        label="Regular (360*n)",
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
                        label="Irregular congruent",
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
                        label="Irregular non-congruent",
                    )
            self.res_status_var.set(
                f"Displayed {np.count_nonzero(mask)} points: {lo:.9g} to {hi:.9g}."
            )
        else:
            ax_iq.text(0.5, 0.5, "Select a wider frequency region.", ha="center", va="center")
            self.res_status_var.set("Selection too small. Drag a wider region.")

        iq_label = "normalized" if use_corrected else "raw"
        ax_iq.set_xlabel(f"Re({iq_label} S21)")
        ax_iq.set_ylabel(f"Im({iq_label} S21)")
        ax_iq.grid(True, alpha=0.3)
        ax_iq.set_title("Complex Plane (Displayed Frequency Window)", fontsize=10)
        ax_iq.set_aspect("equal", adjustable="box")

        ax_amp.legend(loc="best", fontsize=8)
        ax_iq.legend(loc="best", fontsize=8)

        self.res_span_selector = SpanSelector(
            ax_amp,
            self._res_on_select,
            "horizontal",
            useblit=True,
            interactive=True,
            drag_from_anywhere=True,
        )
        self.res_figure.tight_layout()
        self.res_canvas.draw_idle()
        self._res_save_view_settings()

    def _res_on_select(self, xmin: float, xmax: float) -> None:
        self._res_selected_range = (float(xmin), float(xmax))
        self._res_manual_ylim = None
        self._res_save_view_settings()
        self._res_render()

    def _res_close(self) -> None:
        if self.res_window is not None and self.res_window.winfo_exists():
            self.res_window.destroy()
        self.res_window = None
        self.res_canvas = None
        self.res_toolbar = None
        self.res_figure = None
        self.res_status_var = None
        self.res_auto_y_var = None
        self.res_use_corrected_var = None
        self.res_show_phase_var = None
        self.res_span_selector = None
        self.res_amp_ax = None
        self.res_iq_ax = None
        self._res_scan_key = None
        self._res_selected_range = None
        self._res_manual_ylim = None
    def _res_get_raw_complex(self, scan) -> np.ndarray:
        return scan.complex_s21()

    def _res_get_raw_phase(self, scan) -> np.ndarray:
        return np.degrees(np.angle(self._res_get_raw_complex(scan)))
