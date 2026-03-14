from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from tkinter import messagebox

from analysis_filters import _estimate_frequency_resolution_mhz, _window_width_in_freq_units
from analysis_models import _current_user, _make_event

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


def _gaussian_fft_convolve(y: np.ndarray, dx_freq_units: float, width_freq_units: float) -> np.ndarray:
    if width_freq_units <= 0 or dx_freq_units <= 0 or y.size < 3:
        return np.asarray(y, dtype=float).copy()
    sigma_samples = width_freq_units / dx_freq_units
    if sigma_samples < 1e-6:
        return np.asarray(y, dtype=float).copy()
    half = int(max(3, np.ceil(6.0 * sigma_samples)))
    x = np.arange(-half, half + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma_samples) ** 2)
    kernel /= np.sum(kernel)
    yin = _fill_nans_linear(np.asarray(y, dtype=float))
    p = half
    # Reflect-pad before convolution to suppress endpoint artifacts.
    yin_pad = np.pad(yin, (p, p), mode="reflect")
    n = yin_pad.size + kernel.size - 1
    nfft = 1 << int(np.ceil(np.log2(max(2, n))))
    yf = np.fft.rfft(yin_pad, n=nfft)
    kf = np.fft.rfft(kernel, n=nfft)
    conv_full = np.fft.irfft(yf * kf, n=nfft)[:n]
    start = kernel.size // 2
    same_pad = conv_full[start : start + yin_pad.size]
    return same_pad[p : p + yin.size]


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for i, val in enumerate(mask):
        if val and not in_run:
            start = i
            in_run = True
        elif not val and in_run:
            runs.append((start, i - 1))
            in_run = False
    if in_run:
        runs.append((start, len(mask) - 1))
    return runs


class DSDFConvolutionMixin:
    def _dsdf_autoscale_y_for_visible_x(self, ax) -> None:
        if self.dsdf_auto_y_var is None or not bool(self.dsdf_auto_y_var.get()):
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

    def open_dsdf_convolution_window(self) -> None:
        if not self._selected_scans_have_attached_normalized():
            messagebox.showwarning(
                "Missing normalized data",
                "All selected scans must have attached normalized data first.",
            )
            return
        if self.dsdf_window is not None and self.dsdf_window.winfo_exists():
            self.dsdf_window.lift()
            return

        scans = self._selected_scans()
        resolution_mhz = _estimate_frequency_resolution_mhz(scans)
        max_khz = 100.0
        min_khz = min(max(resolution_mhz * 1000.0, 1e-3), max_khz)
        default_khz = min(max(15.0, min_khz), max_khz)
        default_thresh = 0.1
        default_min_region_khz = default_khz

        self.dsdf_window = tk.Toplevel(self.root)
        self.dsdf_window.title("Gaussian Convolution on |dS21/df|")
        self.dsdf_window.geometry("1200x820")
        self.dsdf_window.protocol("WM_DELETE_WINDOW", self._dsdf_close)

        controls = tk.Frame(self.dsdf_window, padx=8, pady=8)
        controls.pack(side="top", fill="x")
        self.dsdf_fwhm_slider = tk.Scale(
            controls,
            from_=min_khz,
            to=max_khz,
            resolution=min_khz,
            orient="horizontal",
            label="Gaussian FWHM (kHz)",
            command=lambda _v: self._dsdf_on_slider_changed(),
            length=320,
        )
        self.dsdf_fwhm_slider.set(default_khz)
        self.dsdf_fwhm_slider.pack(side="left", padx=(0, 12))
        self.dsdf_fwhm_slider.bind("<ButtonRelease-1>", self._dsdf_on_slider_released)
        self.dsdf_fwhm_slider.bind("<KeyRelease>", self._dsdf_on_slider_released)

        self.dsdf_threshold_slider = tk.Scale(
            controls,
            from_=0.0,
            to=0.2,
            resolution=0.01,
            orient="horizontal",
            label="Threshold",
            command=lambda _v: self._dsdf_on_slider_changed(),
            length=220,
        )
        self.dsdf_threshold_slider.set(default_thresh)
        self.dsdf_threshold_slider.pack(side="left", padx=(0, 12))
        self.dsdf_threshold_slider.bind("<ButtonRelease-1>", self._dsdf_on_slider_released)
        self.dsdf_threshold_slider.bind("<KeyRelease>", self._dsdf_on_slider_released)

        self.dsdf_min_region_slider = tk.Scale(
            controls,
            from_=min_khz,
            to=max_khz,
            resolution=min_khz,
            orient="horizontal",
            label="Min Region Width (kHz)",
            command=lambda _v: self._dsdf_on_slider_changed(),
            length=280,
        )
        self.dsdf_min_region_slider.set(default_min_region_khz)
        self.dsdf_min_region_slider.pack(side="left", padx=(0, 12))
        self.dsdf_min_region_slider.bind("<ButtonRelease-1>", self._dsdf_on_slider_released)
        self.dsdf_min_region_slider.bind("<KeyRelease>", self._dsdf_on_slider_released)

        self.dsdf_auto_y_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            controls,
            text="Auto-scale |dS21/df| in window",
            variable=self.dsdf_auto_y_var,
            command=self._dsdf_on_auto_y_toggled,
        ).pack(side="left", padx=(0, 12))
        self.dsdf_show_phase_context_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            controls,
            text="Context: Phase (else |S21|)",
            variable=self.dsdf_show_phase_context_var,
            command=self._dsdf_on_slider_released,
        ).pack(side="left", padx=(0, 12))
        self.dsdf_use_corrected_context_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            controls,
            text="Context: Corrected (else Raw)",
            variable=self.dsdf_use_corrected_context_var,
            command=self._dsdf_on_slider_released,
            state="disabled",
        ).pack(side="left", padx=(0, 12))

        self.dsdf_status_var = tk.StringVar(value="Adjust settings and release slider to update.")
        tk.Label(controls, textvariable=self.dsdf_status_var, anchor="w").pack(
            side="left", fill="x", expand=True
        )

        actions = tk.Frame(self.dsdf_window, padx=8, pady=6)
        actions.pack(side="top", fill="x")
        tk.Button(actions, text="Close", width=12, command=self._dsdf_close).pack(side="right")
        tk.Button(actions, text="Reset View", width=12, command=self._dsdf_reset_view).pack(
            side="right", padx=(8, 0)
        )
        self.dsdf_attach_button = tk.Button(actions, text="Attach", width=12, command=self._dsdf_attach)
        self.dsdf_attach_button.pack(side="right", padx=(8, 0))
        self._dsdf_set_attach_state(attached=False)

        self.dsdf_figure = Figure(figsize=(12, 7))
        self.dsdf_canvas = FigureCanvasTkAgg(self.dsdf_figure, master=self.dsdf_window)
        self.dsdf_toolbar = NavigationToolbar2Tk(self.dsdf_canvas, self.dsdf_window)
        self.dsdf_toolbar.update()
        self.dsdf_toolbar.pack(side="top", fill="x")
        self.dsdf_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.dsdf_canvas.mpl_connect("button_release_event", lambda _e: self._dsdf_autoscale_all_y())

        self.dsdf_preview = {}
        self._dsdf_compute_preview()
        self._dsdf_render()
        self._log(f"Opened |dS21/df| convolution window. Slider range {min_khz:.3f}-{max_khz:.3f} kHz.")

    def _dsdf_set_attach_state(self, attached: bool) -> None:
        if self.dsdf_attach_button is None:
            return
        self.dsdf_attach_button.configure(
            bg="light green" if attached else "pink",
            activebackground="light green" if attached else "pink",
        )

    def _dsdf_on_slider_changed(self) -> None:
        self._dsdf_set_attach_state(attached=False)
        if self.dsdf_status_var is not None:
            self.dsdf_status_var.set("Adjusting settings...")

    def _dsdf_on_slider_released(self, _event: tk.Event | None = None) -> None:
        self._dsdf_compute_preview()
        self._dsdf_render()
        self._dsdf_set_attach_state(attached=False)
        if self.dsdf_status_var is not None:
            self.dsdf_status_var.set("Preview updated. Attach to save.")

    def _dsdf_on_auto_y_toggled(self) -> None:
        if self.dsdf_auto_y_var is not None and bool(self.dsdf_auto_y_var.get()):
            self._dsdf_autoscale_all_y()

    def _dsdf_autoscale_all_y(self) -> None:
        if self.dsdf_figure is None or self.dsdf_canvas is None:
            return
        if self.dsdf_auto_y_var is None or not bool(self.dsdf_auto_y_var.get()):
            return
        for ax in self.dsdf_figure.axes:
            self._dsdf_autoscale_y_for_visible_x(ax)
        self.dsdf_canvas.draw_idle()

    def _dsdf_reset_view(self) -> None:
        if self.dsdf_figure is None or self.dsdf_canvas is None:
            return
        for ax in self.dsdf_figure.axes:
            ax.relim()
            ax.autoscale(enable=True, axis="both", tight=False)
        self.dsdf_canvas.draw_idle()

    def _dsdf_compute_preview(self) -> None:
        self.dsdf_preview = {}
        scans = self._selected_scans()
        if (
            self.dsdf_fwhm_slider is None
            or self.dsdf_threshold_slider is None
            or self.dsdf_min_region_slider is None
        ):
            return
        fwhm_ghz = float(self.dsdf_fwhm_slider.get()) / 1e6
        sigma_ghz = fwhm_ghz * _FWHM_TO_SIGMA
        threshold = float(self.dsdf_threshold_slider.get())
        min_region_width_ghz = float(self.dsdf_min_region_slider.get()) / 1e6

        for scan in scans:
            norm = scan.baseline_filter.get("normalized", {})
            if not isinstance(norm, dict):
                continue
            z = np.asarray(norm.get("norm_complex"), dtype=np.complex128)
            if z.shape != scan.freq.shape:
                continue
            amp = np.abs(z)
            freq = np.asarray(scan.freq, dtype=float)
            order = np.argsort(freq)
            f_sorted = freq[order]
            z_sorted = z[order]
            dz_df_sorted = np.gradient(z_sorted, f_sorted)
            dmag_sorted = np.abs(dz_df_sorted)
            dmag = np.empty_like(dmag_sorted)
            dmag[order] = dmag_sorted
            maxv = float(np.nanmax(dmag)) if np.any(np.isfinite(dmag)) else 0.0
            dmag_norm = dmag / maxv if maxv > 0 else np.zeros_like(dmag)

            diffs = np.diff(np.sort(freq))
            diffs = np.abs(diffs[np.isfinite(diffs)])
            diffs = diffs[diffs > 0]
            dx = float(np.median(diffs)) if diffs.size else 0.0
            width_freq_units = float(_window_width_in_freq_units(freq, sigma_ghz))
            smooth = _gaussian_fft_convolve(dmag_norm, dx_freq_units=dx, width_freq_units=width_freq_units)

            min_region_width_freq = float(_window_width_in_freq_units(freq, min_region_width_ghz))
            above = smooth > threshold
            accepted_regions: list[tuple[int, int]] = []
            maxima_idx: list[int] = []
            for i0, i1 in _true_runs(above):
                if i1 <= i0:
                    continue
                if float(freq[i1] - freq[i0]) < min_region_width_freq:
                    continue
                accepted_regions.append((i0, i1))
                local = smooth[i0 : i1 + 1]
                maxima_idx.append(i0 + int(np.argmax(local)))

            self.dsdf_preview[self._scan_key(scan)] = {
                "context_amp_corr": np.asarray(amp, dtype=float),
                "context_phase_corr_wrapped": np.angle(z),
                "raw_dmag_norm": dmag_norm,
                "smooth_dmag_norm": smooth,
                "fwhm_ghz": fwhm_ghz,
                "sigma_ghz": sigma_ghz,
                "threshold": threshold,
                "min_region_width_ghz": min_region_width_ghz,
                "accepted_regions": accepted_regions,
                "maxima_idx": np.asarray(maxima_idx, dtype=int),
            }

    def _dsdf_render(self) -> None:
        if self.dsdf_figure is None or self.dsdf_canvas is None:
            return
        saved = [(ax.get_xlim(), ax.get_ylim()) for ax in self.dsdf_figure.axes]
        scans = self._selected_scans()
        self.dsdf_figure.clear()
        if not scans:
            ax = self.dsdf_figure.add_subplot(111)
            ax.text(0.5, 0.5, "No selected scans", ha="center", va="center")
            ax.axis("off")
            self.dsdf_canvas.draw_idle()
            return
        n = len(scans)
        axes = np.atleast_1d(self.dsdf_figure.subplots(n, 1, sharex=False))
        for i, scan in enumerate(scans):
            ax = axes[i]
            prev = self.dsdf_preview.get(self._scan_key(scan))
            if prev is None:
                ax.text(0.5, 0.5, "Missing normalized data", ha="center", va="center")
                ax.axis("off")
                continue
            freq = scan.freq
            raw = prev["raw_dmag_norm"]
            sm = prev["smooth_dmag_norm"]
            show_phase = bool(self.dsdf_show_phase_context_var.get()) if self.dsdf_show_phase_context_var is not None else False
            if show_phase:
                ctx = prev["context_phase_corr_wrapped"]
                ctx_label = "Corrected phase (wrapped, rad) context"
            else:
                ctx = prev["context_amp_corr"]
                ctx_label = "Corrected |S21| context"
            ax.plot(
                freq,
                ctx,
                color="0.85",
                linewidth=1.0,
                label=ctx_label,
                zorder=0,
            )
            ax.plot(freq, raw, color="0.5", linewidth=0.8, label="Raw |dS21/df| (norm)")
            ax.plot(freq, sm, color="tab:blue", linestyle="--", linewidth=1.2, label="Smoothed (non-accepted)")
            for r0, r1 in prev["accepted_regions"]:
                ax.plot(
                    freq[r0 : r1 + 1], sm[r0 : r1 + 1], color="tab:green", linewidth=1.5,
                    label="Accepted region" if (r0, r1) == prev["accepted_regions"][0] else None,
                )
            idx = prev["maxima_idx"]
            if idx.size:
                ax.plot(freq[idx], raw[idx], "o", color="tab:red", markersize=4, label="Region maxima (raw)")
                ax.plot(freq[idx], sm[idx], "x", color="black", markersize=5, label="Region maxima (smooth)")
            if show_phase:
                ax.set_ylabel("|dS21/df| (norm) + Phase context (rad)")
            else:
                ax.set_ylabel("|dS21/df| (norm) + |S21| context")
            ax.grid(True, alpha=0.3)
            ax.set_title(Path(scan.filename).name, fontsize=9)
            if i == 0:
                ax.legend(loc="upper right", fontsize=8)
            if i == n - 1:
                ax.set_xlabel("Frequency")
        fwhm_khz = float(self.dsdf_fwhm_slider.get()) if self.dsdf_fwhm_slider else 0.0
        th = float(self.dsdf_threshold_slider.get()) if self.dsdf_threshold_slider else 0.0
        w_khz = float(self.dsdf_min_region_slider.get()) if self.dsdf_min_region_slider else 0.0
        self.dsdf_figure.suptitle(
            f"|dS21/df| Gaussian Convolution | FWHM={fwhm_khz:.3f} kHz | threshold={th:.2f} | min width={w_khz:.3f} kHz",
            fontsize=11,
        )
        if len(saved) == len(axes):
            for ax, (xlim, ylim) in zip(axes, saved):
                ax.set_xlim(xlim)
                if self.dsdf_auto_y_var is not None and bool(self.dsdf_auto_y_var.get()):
                    self._dsdf_autoscale_y_for_visible_x(ax)
                else:
                    ax.set_ylim(ylim)
        for ax in axes:
            ax.callbacks.connect("xlim_changed", lambda changed_ax: self._dsdf_autoscale_y_for_visible_x(changed_ax))
        self.dsdf_figure.tight_layout()
        self.dsdf_canvas.draw_idle()

    def _dsdf_attach(self) -> None:
        scans = self._selected_scans()
        if not scans:
            return
        attached_at = datetime.now().isoformat(timespec="seconds")
        count = 0
        for scan in scans:
            prev = self.dsdf_preview.get(self._scan_key(scan))
            if prev is None:
                continue
            norm = scan.baseline_filter.get("normalized", {})
            if not isinstance(norm, dict):
                continue
            norm["dsdf_conv"] = {
                "attached_at": attached_at,
                "attached_by": _current_user(),
                "fwhm_ghz": prev["fwhm_ghz"],
                "sigma_ghz": prev["sigma_ghz"],
                "threshold": prev["threshold"],
                "min_region_width_ghz": prev["min_region_width_ghz"],
                "accepted_regions_indices": prev["accepted_regions"],
                "maxima_indices": prev["maxima_idx"],
                "maxima_freq": scan.freq[prev["maxima_idx"]],
                "smooth_dmag_norm": prev["smooth_dmag_norm"],
                "raw_dmag_norm": prev["raw_dmag_norm"],
            }
            scan.candidate_resonators["dsdf_gaussian_convolution"] = {
                "attached_at": attached_at,
                "attached_by": _current_user(),
                "method": "dsdf_gaussian_convolution",
                "candidate_freq": np.asarray(scan.freq[prev["maxima_idx"]], dtype=float),
                "candidate_indices": np.asarray(prev["maxima_idx"], dtype=int),
                "threshold": prev["threshold"],
                "min_region_width_ghz": prev["min_region_width_ghz"],
                "gaussian_fwhm_ghz": prev["fwhm_ghz"],
            }
            scan.processing_history.append(
                _make_event(
                    "attach_dsdf_convolution",
                    {
                        "candidate_count": int(len(prev["maxima_idx"])),
                        "threshold": prev["threshold"],
                        "fwhm_ghz": prev["fwhm_ghz"],
                    },
                )
            )
            count += 1
        self.dataset.processing_history.append(
            _make_event("attach_dsdf_convolution_selected", {"selected_count": count})
        )
        self._mark_dirty()
        self._refresh_status()
        self._dsdf_set_attach_state(attached=True)
        if self.dsdf_status_var is not None:
            self.dsdf_status_var.set(f"Attached to {count} selected scan(s).")
        self._log(f"Attached |dS21/df| convolution data to {count} selected scan(s).")
        self._autosave_dataset()

    def _dsdf_close(self) -> None:
        if self.dsdf_window is not None and self.dsdf_window.winfo_exists():
            self.dsdf_window.destroy()
        self.dsdf_window = None
        self.dsdf_canvas = None
        self.dsdf_toolbar = None
        self.dsdf_figure = None
        self.dsdf_fwhm_slider = None
        self.dsdf_threshold_slider = None
        self.dsdf_min_region_slider = None
        self.dsdf_auto_y_var = None
        self.dsdf_show_phase_context_var = None
        self.dsdf_use_corrected_context_var = None
        self.dsdf_status_var = None
        self.dsdf_attach_button = None
        self.dsdf_preview = {}
