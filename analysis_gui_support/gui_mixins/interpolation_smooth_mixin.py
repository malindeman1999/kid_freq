from __future__ import annotations

from datetime import datetime
from typing import Dict, List

import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from tkinter import messagebox

from ..analysis_filters import _estimate_frequency_resolution_mhz, _window_width_in_freq_units
from ..analysis_models import _current_user, _make_event, _read_polar_series

_FWHM_TO_SIGMA = 1.0 / 2.3548200450309493


def _fill_nans_linear(y: np.ndarray) -> np.ndarray:
    out = np.asarray(y, dtype=float).copy()
    good = np.isfinite(out)
    if np.all(good):
        return out
    if not np.any(good):
        return np.zeros_like(out)
    idx = np.arange(out.size)
    out[~good] = np.interp(idx[~good], idx[good], out[good])
    return out


def _gaussian_fft_convolve(freq: np.ndarray, y: np.ndarray, fwhm_ghz: float) -> np.ndarray:
    width_freq_units = float(_window_width_in_freq_units(freq, fwhm_ghz * _FWHM_TO_SIGMA))
    if width_freq_units <= 0 or y.size < 3:
        return np.asarray(y, dtype=float).copy()

    f_sorted = np.sort(np.asarray(freq, dtype=float))
    diffs = np.diff(f_sorted)
    diffs = diffs[np.isfinite(diffs)]
    diffs = np.abs(diffs[diffs > 0])
    dx = float(np.median(diffs)) if diffs.size else 0.0
    if dx <= 0:
        return np.asarray(y, dtype=float).copy()

    sigma_samples = width_freq_units / dx
    if sigma_samples < 1e-6:
        return np.asarray(y, dtype=float).copy()

    half = int(max(3, np.ceil(6.0 * sigma_samples)))
    x = np.arange(-half, half + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma_samples) ** 2)
    kernel /= np.sum(kernel)

    yin = _fill_nans_linear(np.asarray(y, dtype=float))
    p = half
    yin_pad = np.pad(yin, (p, p), mode="reflect")
    n = yin_pad.size + kernel.size - 1
    nfft = 1 << int(np.ceil(np.log2(max(2, n))))
    yf = np.fft.rfft(yin_pad, n=nfft)
    kf = np.fft.rfft(kernel, n=nfft)
    conv_full = np.fft.irfft(yf * kf, n=nfft)[:n]
    start = kernel.size // 2
    same_pad = conv_full[start : start + yin_pad.size]
    return same_pad[p : p + yin.size]


class InterpolationSmoothMixin:
    def _interp_autoscale_y_for_visible_x(self, ax) -> None:
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

    def open_interp_smooth_window(self) -> None:
        if not self._selected_scans_have_attached_filter():
            messagebox.showwarning(
                "Missing filter data",
                "Run pipeline in order:\n"
                "Phase Correction -> Baseline Filtering -> Interp+Smooth.\n\n"
                "All selected scans must have attached baseline-filter data first.",
            )
            return

        if self.interp_window is not None and self.interp_window.winfo_exists():
            self.interp_window.lift()
            return

        scans = self._selected_scans()
        self.interp_window = tk.Toplevel(self.root)
        self.interp_window.title("Interpolation + Smoothing")
        self.interp_window.geometry("1200x820")
        self.interp_window.protocol("WM_DELETE_WINDOW", self._interp_close)

        controls = tk.Frame(self.interp_window, padx=8, pady=8)
        controls.pack(side="top", fill="x")

        resolution_mhz = _estimate_frequency_resolution_mhz(scans)
        min_mhz = min(max(resolution_mhz, 1e-6), 50.0)
        max_mhz = 50.0
        default_smooth_mhz = min(max(25.0, min_mhz), max_mhz)

        saved_widths_ghz = []
        for scan in scans:
            bf = scan.baseline_filter
            if not isinstance(bf, dict):
                continue
            interp = bf.get("interp_smooth")
            if not isinstance(interp, dict):
                continue
            if "smoothing_width_ghz" not in interp:
                continue
            saved_widths_ghz.append(float(interp["smoothing_width_ghz"]))

        if saved_widths_ghz:
            uniq = list(dict.fromkeys(saved_widths_ghz))
            chosen_ghz = None
            if len(uniq) > 1:
                labels = [
                    f"{idx+1}. smoothing width={w*1000.0:.3f} MHz"
                    for idx, w in enumerate(uniq)
                ]
                pick = self._select_setting_option(
                    "Interp + Smooth Setting",
                    "Selected scans have different saved smoothing settings. Choose defaults:",
                    labels,
                )
                if pick is not None:
                    chosen_ghz = uniq[pick]
                    self._log(f"Loaded chosen saved interp/smooth setting #{pick + 1} into defaults.")
            else:
                chosen_ghz = uniq[0]
                self._log("Loaded saved interp/smooth setting into defaults.")

            if chosen_ghz is not None:
                default_smooth_mhz = min(max(chosen_ghz * 1000.0, min_mhz), max_mhz)

        self.interp_smooth_slider = tk.Scale(
            controls,
            from_=min_mhz,
            to=max_mhz,
            resolution=min_mhz,
            orient="horizontal",
            label="Smoothing Width (MHz)",
            command=lambda _v: self._interp_on_slider_changed(),
            length=380,
        )
        self.interp_smooth_slider.set(default_smooth_mhz)
        self.interp_smooth_slider.pack(side="left", padx=(0, 12))
        self.interp_smooth_slider.bind("<ButtonRelease-1>", self._interp_on_slider_released)
        self.interp_smooth_slider.bind("<KeyRelease>", self._interp_on_slider_released)

        self.interp_status_var = tk.StringVar(value="Adjust smoothing and release to update.")
        tk.Label(controls, textvariable=self.interp_status_var, anchor="w").pack(
            side="left", fill="x", expand=True
        )

        action_frame = tk.Frame(self.interp_window, padx=8, pady=6)
        action_frame.pack(side="top", fill="x")
        tk.Button(action_frame, text="Cancel", width=12, command=self._interp_close).pack(side="right")
        tk.Button(
            action_frame,
            text="Reset View",
            width=12,
            command=self._interp_reset_view,
        ).pack(side="right", padx=(8, 0))
        self.interp_attach_button = tk.Button(
            action_frame,
            text="Attach, Save, and Close",
            width=24,
            command=self._attach_save_and_close_interp,
        )
        self.interp_attach_button.pack(side="right", padx=(8, 0))
        self._interp_set_attach_state(attached=False)

        self.interp_figure = Figure(figsize=(12, 7))
        self.interp_canvas = FigureCanvasTkAgg(self.interp_figure, master=self.interp_window)
        self.interp_toolbar = NavigationToolbar2Tk(self.interp_canvas, self.interp_window)
        self.interp_toolbar.update()
        def _home_interp(*_args) -> None:
            if self.interp_figure is None or self.interp_canvas is None:
                return
            for ax in self.interp_figure.axes:
                ax.relim()
                ax.autoscale(enable=True, axis="both", tight=False)
            self.interp_canvas.draw_idle()
        self.interp_toolbar.home = _home_interp
        self.interp_toolbar.pack(side="top", fill="x")
        self.interp_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.interp_canvas.mpl_connect(
            "button_release_event", lambda _evt: self._interp_autoscale_all_y()
        )

        self.interp_preview = {}
        self._interp_compute_preview()
        self._interp_render()
        self._log(
            f"Opened interpolation/smoothing window. Slider range {min_mhz:.6f}-{max_mhz:.3f} MHz."
        )

    def _interp_set_attach_state(self, attached: bool) -> None:
        if self.interp_attach_button is None:
            return
        if attached:
            self.interp_attach_button.configure(bg="light green", activebackground="light green")
        else:
            self.interp_attach_button.configure(bg="pink", activebackground="pink")

    def _interp_on_slider_changed(self) -> None:
        if self.interp_status_var is not None:
            self.interp_status_var.set("Adjusting smoothing...")
        self._interp_set_attach_state(attached=False)

    def _interp_on_slider_released(self, _event: tk.Event) -> None:
        self._interp_compute_preview()
        self._interp_render()
        self._interp_set_attach_state(attached=False)
        if self.interp_status_var is not None:
            self.interp_status_var.set("Preview updated. Attach to save.")

    def _interp_autoscale_all_y(self) -> None:
        if self.interp_figure is None or self.interp_canvas is None:
            return
        for ax in self.interp_figure.axes:
            self._interp_autoscale_y_for_visible_x(ax)
        self.interp_canvas.draw_idle()

    def _interp_reset_view(self) -> None:
        if self.interp_figure is None or self.interp_canvas is None:
            return
        for ax in self.interp_figure.axes:
            ax.relim()
            ax.autoscale(enable=True, axis="both", tight=False)
        self.interp_canvas.draw_idle()

    def _interp_compute_preview(self) -> None:
        scans = self._selected_scans()
        self.interp_preview = {}
        if self.interp_smooth_slider is None:
            return
        width_ghz = float(self.interp_smooth_slider.get()) / 1000.0
        for scan in scans:
            bf = scan.baseline_filter
            filtered_amp = np.asarray(bf.get("filtered_amp"), dtype=float)
            filtered_phase = np.asarray(bf.get("filtered_phase_deg"), dtype=float)
            keep = np.asarray(bf.get("retained_mask"), dtype=bool)
            if filtered_amp.ndim != 1 or filtered_phase.ndim != 1 or filtered_amp.shape != filtered_phase.shape:
                continue
            if filtered_amp.size == 0:
                continue
            if keep.shape != scan.freq.shape or np.count_nonzero(keep) != filtered_amp.size:
                continue
            f_keep = np.asarray(scan.freq[keep], dtype=float)
            a_keep = np.asarray(filtered_amp, dtype=float)
            p_keep = np.asarray(filtered_phase, dtype=float)

            order = np.argsort(f_keep)
            f_keep = f_keep[order]
            a_keep = a_keep[order]
            p_keep = p_keep[order]

            f_full = scan.freq
            a_interp = np.interp(f_full, f_keep, a_keep)
            p_interp = np.interp(f_full, f_keep, p_keep)

            a_smooth = _gaussian_fft_convolve(f_full, a_interp, width_ghz)
            p_smooth = _gaussian_fft_convolve(f_full, p_interp, width_ghz)
            self.interp_preview[self._scan_key(scan)] = {
                "interp_amp": a_interp,
                "interp_phase": p_interp,
                "smooth_amp": a_smooth,
                "smooth_phase": p_smooth,
                "source_points_freq": f_keep,
                "source_points_amp": a_keep,
                "source_points_phase": p_keep,
                "smoothing_width_ghz": width_ghz,
            }

    def _interp_render(self) -> None:
        if self.interp_figure is None or self.interp_canvas is None:
            return
        saved_x_limits = []
        for ax_old in self.interp_figure.axes:
            saved_x_limits.append(ax_old.get_xlim())
        scans = self._selected_scans()
        if not scans:
            self.interp_figure.clear()
            self.interp_canvas.draw_idle()
            return

        n = len(scans)
        self.interp_figure.clear()
        axes = self.interp_figure.subplots(n, 2, sharex=False)
        axes = np.atleast_2d(axes)

        for i, scan in enumerate(scans):
            freq_ghz = np.asarray(scan.freq, dtype=float) / 1.0e9
            ax_a = axes[i, 0]
            ax_p = axes[i, 1]
            prev = self.interp_preview.get(self._scan_key(scan))
            if prev is None:
                continue
            bf = scan.baseline_filter if isinstance(scan.baseline_filter, dict) else {}
            keep = np.asarray(bf.get("retained_mask"), dtype=bool)
            phase3 = scan.candidate_resonators.get("phase_correction_3", {})
            amp_input, phase_input = _read_polar_series(
                phase3,
                amplitude_key="corrected_amp",
                phase_key="corrected_phase_deg",
            )

            interp_amp = np.asarray(prev["interp_amp"], dtype=float)
            interp_phase = np.asarray(prev["interp_phase"], dtype=float)
            smooth_amp = np.asarray(prev["smooth_amp"], dtype=float)
            smooth_phase = np.asarray(prev["smooth_phase"], dtype=float)

            ax_a.plot(
                freq_ghz,
                amp_input,
                color="0.6",
                linewidth=0.8,
                label="Amplitude input",
            )
            ax_a.plot(freq_ghz, interp_amp, color="tab:blue", linewidth=0.8, label="Interp")
            if keep.shape == scan.freq.shape and np.any(keep):
                ax_a.plot(
                    freq_ghz[keep],
                    amp_input[keep],
                    linestyle="none",
                    marker=".",
                    markersize=2.5,
                    color="tab:orange",
                    label="Retained",
                )
            ax_a.plot(freq_ghz, smooth_amp, color="red", linewidth=2.0, label="Smoothed", zorder=4)
            ax_a.set_ylabel("|S21|")
            ax_a.grid(True, alpha=0.3)
            ax_a.set_title(scan.filename.split("\\")[-1], fontsize=9)

            ax_p.plot(
                freq_ghz,
                phase_input,
                color="0.6",
                linewidth=0.8,
                label="Phase input",
            )
            ax_p.plot(
                freq_ghz,
                interp_phase,
                color="tab:blue",
                linewidth=0.8,
                label="Interp",
            )
            if keep.shape == scan.freq.shape and np.any(keep):
                ax_p.plot(
                    freq_ghz[keep],
                    phase_input[keep],
                    linestyle="none",
                    marker=".",
                    markersize=2.5,
                    color="tab:orange",
                    label="Retained-associated phase",
                )
            ax_p.plot(
                freq_ghz,
                smooth_phase,
                color="red",
                linewidth=2.0,
                label="Smoothed",
                zorder=4,
            )
            ax_p.set_ylabel("Phase (deg)")
            ax_p.grid(True, alpha=0.3)

            if i == 0:
                ax_a.legend(loc="upper right", fontsize=8)
                ax_p.legend(loc="upper right", fontsize=8)
            if i == n - 1:
                ax_a.set_xlabel("Frequency (GHz)")
                ax_p.set_xlabel("Frequency (GHz)")

        width_mhz = float(self.interp_smooth_slider.get()) if self.interp_smooth_slider else 0.0
        self.interp_figure.suptitle(
            f"Interpolated + Smoothed (width={width_mhz:.3f} MHz)", fontsize=11
        )
        new_axes = list(axes.ravel())
        if len(saved_x_limits) == len(new_axes):
            for ax_new, xlim in zip(new_axes, saved_x_limits):
                ax_new.set_xlim(xlim)
                self._interp_autoscale_y_for_visible_x(ax_new)
        for ax_new in new_axes:
            ax_new.callbacks.connect(
                "xlim_changed",
                lambda changed_ax: self._interp_autoscale_y_for_visible_x(changed_ax),
            )
        self.interp_figure.tight_layout()
        self.interp_canvas.draw_idle()

    def _interp_attach(self) -> None:
        scans = self._selected_scans()
        if not scans:
            return
        attached_at = datetime.now().isoformat(timespec="seconds")
        count = 0
        for scan in scans:
            key = self._scan_key(scan)
            prev = self.interp_preview.get(key)
            if prev is None:
                continue
            # Exactly one attached interp/smooth result per scan; overwrite prior attachment.
            scan.baseline_filter["interp_smooth"] = {}
            scan.baseline_filter["interp_smooth"] = {
                "attached_at": attached_at,
                "attached_by": _current_user(),
                "smoothing_width_ghz": prev["smoothing_width_ghz"],
                "smoothing_width_mhz": prev["smoothing_width_ghz"] * 1000.0,
                "interp_amp": np.asarray(prev["interp_amp"], dtype=float),
                "interp_phase": np.asarray(prev["interp_phase"], dtype=float),
                "interp_data_polar": np.vstack((scan.freq, prev["interp_amp"], prev["interp_phase"])),
                "interp_data_polar_format": "(3, N) rows = [freq, amplitude, unwrapped_phase_deg]",
                "smooth_amp": np.asarray(prev["smooth_amp"], dtype=float),
                "smooth_phase": np.asarray(prev["smooth_phase"], dtype=float),
                "smooth_data_polar": np.vstack((scan.freq, prev["smooth_amp"], prev["smooth_phase"])),
                "smooth_data_polar_format": "(3, N) rows = [freq, amplitude, unwrapped_phase_deg]",
            }
            scan.processing_history.append(
                _make_event(
                    "attach_interp_smooth",
                    {
                        "smoothing_width_ghz": prev["smoothing_width_ghz"],
                        "smoothing_width_mhz": prev["smoothing_width_ghz"] * 1000.0,
                        "points": int(len(scan.freq)),
                    },
                )
            )
            count += 1
        self.dataset.processing_history.append(
            _make_event("attach_interp_smooth_selected", {"selected_count": count})
        )
        self._mark_dirty()
        self._refresh_status()
        self._interp_set_attach_state(attached=True)
        if self.interp_status_var is not None:
            self.interp_status_var.set(f"Attached to {count} selected scan(s).")
        self._log(f"Attached interpolated+smoothed data to {count} selected scan(s).")
        self._autosave_dataset()

    def _interp_close(self) -> None:
        if self.interp_window is not None and self.interp_window.winfo_exists():
            self.interp_window.destroy()
        self.interp_window = None
        self.interp_canvas = None
        self.interp_toolbar = None
        self.interp_figure = None
        self.interp_status_var = None
        self.interp_smooth_slider = None
        self.interp_attach_button = None
        self.interp_preview = {}
