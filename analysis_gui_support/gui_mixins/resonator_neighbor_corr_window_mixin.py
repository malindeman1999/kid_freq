from __future__ import annotations

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

class ResonatorNeighborCorrWindowMixin:
    def open_resonator_neighbor_corr_window(self) -> None:
        if self.res_neighbor_corr_window is not None and self.res_neighbor_corr_window.winfo_exists():
            self.res_neighbor_corr_window.lift()
            self._render_resonator_neighbor_corr_window()
            return

        self.res_neighbor_corr_window = tk.Toplevel(self.root)
        self.res_neighbor_corr_window.title("Neighbor Pair Self Correlation vs Time")
        self.res_neighbor_corr_window.geometry("1380x900")
        self.res_neighbor_corr_window.protocol("WM_DELETE_WINDOW", self._close_resonator_neighbor_corr_window)

        controls = tk.Frame(self.res_neighbor_corr_window, padx=8, pady=8)
        controls.pack(side="top", fill="x")
        control_row = tk.Frame(controls)
        control_row.pack(side="top", fill="x", anchor="w")
        self.res_neighbor_corr_status_var = tk.StringVar(
            value="Showing correlation of each pair-drift interval with the initial interval."
        )
        self.res_neighbor_corr_sep_rel_var = tk.DoubleVar(value=0.004)
        self.res_neighbor_corr_initial_date_var = tk.StringVar(value=self._dataset_res_neighbor_initial_date())
        self.res_neighbor_corr_show_curves_var = tk.BooleanVar(value=True)

        tk.Label(control_row, text="Initial Date").pack(side="left", padx=(0, 4))
        initial_date_entry = tk.Entry(control_row, width=12, textvariable=self.res_neighbor_corr_initial_date_var)
        initial_date_entry.pack(side="left", padx=(0, 4))
        initial_date_entry.bind("<Return>", lambda _event: self._render_resonator_neighbor_corr_window())
        initial_date_entry.bind("<FocusOut>", lambda _event: self._render_resonator_neighbor_corr_window())
        tk.Label(control_row, text="YYYY-MM-DD").pack(side="left", padx=(0, 12))
        self.res_neighbor_corr_sep_scale = tk.Scale(
            control_row,
            from_=0.0,
            to=0.04,
            resolution=0.0001,
            orient="horizontal",
            length=260,
            digits=5,
            label="Max pair mean separation (df/f)",
            variable=self.res_neighbor_corr_sep_rel_var,
            command=lambda _value: self._render_resonator_neighbor_corr_window(),
        )
        self.res_neighbor_corr_sep_scale.pack(side="left", padx=(0, 12))
        tk.Checkbutton(
            control_row,
            text="Colored Curves",
            variable=self.res_neighbor_corr_show_curves_var,
            command=self._render_resonator_neighbor_corr_window,
        ).pack(side="left", padx=(0, 12))
        tk.Button(
            control_row,
            text="Show On Scans",
            width=13,
            command=lambda: self.open_resonator_neighbor_scan_window(source="corr"),
        ).pack(side="left", padx=(0, 8))
        tk.Button(control_row, text="Refresh", width=10, command=self._render_resonator_neighbor_corr_window).pack(
            side="left",
            padx=(0, 8),
        )
        tk.Label(controls, textvariable=self.res_neighbor_corr_status_var, anchor="w", justify="left").pack(
            side="top",
            fill="x",
            expand=True,
            pady=(6, 0),
        )

        self.res_neighbor_corr_figure = Figure(figsize=(12.5, 7.8))
        self.res_neighbor_corr_canvas = FigureCanvasTkAgg(
            self.res_neighbor_corr_figure,
            master=self.res_neighbor_corr_window,
        )
        self.res_neighbor_corr_toolbar = NavigationToolbar2Tk(
            self.res_neighbor_corr_canvas,
            self.res_neighbor_corr_window,
        )
        self.res_neighbor_corr_toolbar.update()
        self.res_neighbor_corr_toolbar.pack(side="top", fill="x")
        self.res_neighbor_corr_canvas.get_tk_widget().pack(fill="both", expand=True)

        self._render_resonator_neighbor_corr_window()



    def _close_resonator_neighbor_corr_window(self) -> None:
        if self.res_neighbor_corr_window is not None and self.res_neighbor_corr_window.winfo_exists():
            self.res_neighbor_corr_window.destroy()
        self.res_neighbor_corr_window = None
        self.res_neighbor_corr_canvas = None
        self.res_neighbor_corr_toolbar = None
        self.res_neighbor_corr_figure = None
        self.res_neighbor_corr_status_var = None
        self.res_neighbor_corr_sep_rel_var = None
        self.res_neighbor_corr_initial_date_var = None
        self.res_neighbor_corr_show_curves_var = None
        self.res_neighbor_corr_sep_scale = None
        self._res_neighbor_corr_axes = None



    def _render_resonator_neighbor_corr_window(self) -> None:
        if self.res_neighbor_corr_figure is None or self.res_neighbor_corr_canvas is None:
            return

        self.res_neighbor_corr_figure.clear()
        ax_corr = self.res_neighbor_corr_figure.add_subplot(211)
        ax_mag = self.res_neighbor_corr_figure.add_subplot(212, sharex=ax_corr)
        self._res_neighbor_corr_axes = (ax_corr, ax_mag)

        threshold_rel = (
            float(self.res_neighbor_corr_sep_rel_var.get())
            if self.res_neighbor_corr_sep_rel_var is not None
            else 0.004
        )
        initial_date_text = (
            str(self.res_neighbor_corr_initial_date_var.get())
            if self.res_neighbor_corr_initial_date_var is not None
            else ""
        )

        try:
            data = self._resonator_neighbor_dfrel_data(threshold_rel, initial_date_text=initial_date_text)
            summary = self._resonator_neighbor_self_correlation_summary(data)
        except Exception as exc:
            ax_corr.text(0.5, 0.5, str(exc), ha="center", va="center", transform=ax_corr.transAxes)
            ax_corr.set_axis_off()
            ax_mag.set_axis_off()
            if self.res_neighbor_corr_status_var is not None:
                self.res_neighbor_corr_status_var.set(str(exc))
            self.res_neighbor_corr_canvas.draw_idle()
            return

        points = summary["points"]
        x_end = np.asarray([float(item["end_elapsed_days"]) for item in points], dtype=float)
        y_corr = np.asarray([float(item["corr_to_initial"]) for item in points], dtype=float)
        q1_corr = np.asarray([float(item["q1_contribution"]) for item in points], dtype=float)
        q3_corr = np.asarray([float(item["q3_contribution"]) for item in points], dtype=float)
        pair_counts = np.asarray([int(item["pair_count"]) for item in points], dtype=int)
        common_counts = np.asarray([int(item["common_pair_count"]) for item in points], dtype=int)
        y_mag = np.asarray([float(item["mean_abs_drift_rate"]) for item in points], dtype=float)
        y_mag_std = np.asarray([float(item["std_abs_drift_rate"]) for item in points], dtype=float)
        norm, cmap = self._resonator_neighbor_pair_colors(data)
        pair_mean_freq_by_label = dict(summary.get("pair_mean_freq_by_label", {}))
        show_curves = (
            bool(self.res_neighbor_corr_show_curves_var.get())
            if self.res_neighbor_corr_show_curves_var is not None
            else True
        )

        contribution_series: dict[str, list[tuple[float, float]]] = {}
        for point in points:
            x_value = float(point["end_elapsed_days"])
            if not np.isfinite(x_value):
                continue
            contribution_by_label = point.get("contribution_by_label", {})
            if not isinstance(contribution_by_label, dict):
                continue
            for pair_label, value in contribution_by_label.items():
                y_value = float(value)
                if not np.isfinite(y_value):
                    continue
                contribution_series.setdefault(str(pair_label), []).append((x_value, y_value))

        q_mask = np.isfinite(x_end) & np.isfinite(q1_corr) & np.isfinite(q3_corr)
        if np.any(q_mask):
            ax_corr.fill_between(
                x_end[q_mask],
                q1_corr[q_mask],
                q3_corr[q_mask],
                color="0.2",
                alpha=0.22,
                linewidth=0.0,
                zorder=1,
                label="Middle 50%",
            )

        if show_curves:
            for pair_label, series in sorted(
                contribution_series.items(),
                key=lambda item: (
                    pair_mean_freq_by_label.get(item[0], np.nan),
                    self._resonator_sort_key(item[0]),
                ),
            ):
                pair_freq_hz = float(pair_mean_freq_by_label.get(pair_label, np.nan))
                color = cmap(norm(pair_freq_hz)) if np.isfinite(pair_freq_hz) else "0.55"
                xy = np.asarray(series, dtype=float)
                if xy.ndim != 2 or xy.shape[0] == 0:
                    continue
                order = np.argsort(xy[:, 0])
                ax_corr.plot(
                    xy[order, 0],
                    xy[order, 1],
                    color=color,
                    linewidth=1.2,
                    alpha=0.9,
                    marker="o",
                    markersize=3.5,
                    zorder=2,
                )

        corr_mask = np.isfinite(x_end) & np.isfinite(y_corr)
        if np.any(corr_mask):
            ax_corr.plot(
                x_end[corr_mask],
                y_corr[corr_mask],
                color="black",
                linewidth=4.0,
                marker="o",
                markersize=5.5,
                zorder=4,
                label="Mean correlation",
            )
        missing_mask = np.isfinite(x_end) & ~np.isfinite(y_corr)
        if np.any(missing_mask):
            ax_corr.plot(
                x_end[missing_mask],
                np.zeros(np.count_nonzero(missing_mask), dtype=float),
                linestyle="none",
                marker="x",
                markersize=6.0,
                color="tab:red",
                alpha=0.8,
                zorder=4,
            )

        ref_idx = int(summary["reference_interval_idx"])
        if 0 <= ref_idx < x_end.size and np.isfinite(x_end[ref_idx]):
            ax_corr.axvline(x_end[ref_idx], color="0.45", linestyle=":", linewidth=1.0, alpha=0.9)

        for x_value, y_value, common_count in zip(x_end, y_corr, common_counts):
            if not np.isfinite(x_value) or not np.isfinite(y_value):
                continue
            ax_corr.annotate(
                f"n={int(common_count)}",
                (float(x_value), float(y_value)),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                alpha=0.85,
            )

        mag_mask = np.isfinite(x_end) & np.isfinite(y_mag)
        if np.any(mag_mask):
            lower = np.maximum(0.0, y_mag[mag_mask] - np.nan_to_num(y_mag_std[mag_mask], nan=0.0))
            upper = y_mag[mag_mask] + np.nan_to_num(y_mag_std[mag_mask], nan=0.0)
            ax_mag.fill_between(
                x_end[mag_mask],
                lower,
                upper,
                color="0.2",
                alpha=0.22,
                linewidth=0.0,
                zorder=1,
                label="Mean |drift| +/- 1 std",
            )
            ax_mag.plot(
                x_end[mag_mask],
                y_mag[mag_mask],
                color="black",
                linewidth=1.8,
                marker="o",
                markersize=5.0,
                zorder=3,
                label="Mean |drift|",
            )

        for x_value, y_value, pair_count in zip(x_end, y_mag, pair_counts):
            if not np.isfinite(x_value) or not np.isfinite(y_value):
                continue
            ax_mag.annotate(
                f"n={int(pair_count)}",
                (float(x_value), float(y_value)),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                alpha=0.85,
            )

        ax_corr.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--")
        ax_corr.set_ylim(-2.0, 2.0)
        ax_corr.set_ylabel("Correlation to Initial Interval")
        ax_corr.set_title("Neighbor Pair Drift Self Correlation")
        ax_corr.grid(True, alpha=0.3)
        if contribution_series:
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            colorbar = self.res_neighbor_corr_figure.colorbar(sm, ax=ax_corr, pad=0.02)
            colorbar.set_label("Pair Mean Frequency (GHz)")
            tick_vals = colorbar.get_ticks()
            tick_vals = [tick for tick in tick_vals if float(norm.vmin) <= float(tick) <= float(norm.vmax)]
            if tick_vals:
                colorbar.set_ticks(tick_vals)
            colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value / 1.0e9:.3f}"))
            vmin = float(norm.vmin)
            vmax = float(norm.vmax)
            freq_span_hz = max(vmax - vmin, 1.0)
            sorted_pair_freqs_hz = np.sort(
                np.asarray(
                    [
                        float(freq_hz)
                        for freq_hz in pair_mean_freq_by_label.values()
                        if np.isfinite(float(freq_hz))
                    ],
                    dtype=float,
                )
            )
            if sorted_pair_freqs_hz.size >= 2:
                neighbor_steps_hz = np.diff(sorted_pair_freqs_hz)
                neighbor_steps_hz = neighbor_steps_hz[np.isfinite(neighbor_steps_hz) & (neighbor_steps_hz > 0.0)]
                band_halfwidth_hz = (
                    0.18 * float(np.min(neighbor_steps_hz))
                    if neighbor_steps_hz.size
                    else 0.012 * freq_span_hz
                )
            else:
                band_halfwidth_hz = 0.018 * freq_span_hz
            band_halfwidth_hz = min(max(band_halfwidth_hz, 0.004 * freq_span_hz), 0.045 * freq_span_hz)

            band_edges: list[tuple[float, float]] = []
            cursor_hz = vmin
            for pair_freq_hz in sorted_pair_freqs_hz:
                lo_hz = max(vmin, float(pair_freq_hz) - band_halfwidth_hz)
                hi_hz = min(vmax, float(pair_freq_hz) + band_halfwidth_hz)
                if lo_hz > cursor_hz:
                    band_edges.append((cursor_hz, lo_hz))
                cursor_hz = max(cursor_hz, hi_hz)
            if cursor_hz < vmax:
                band_edges.append((cursor_hz, vmax))

            for lo_hz, hi_hz in band_edges:
                if hi_hz <= lo_hz:
                    continue
                mid_hz = 0.5 * (lo_hz + hi_hz)
                base_color = np.asarray(cmap(norm(float(mid_hz)))[:3], dtype=float)
                dark_color = tuple(np.clip(0.55 * base_color, 0.0, 1.0))
                colorbar.ax.axhspan(
                    lo_hz,
                    hi_hz,
                    xmin=0.0,
                    xmax=1.0,
                    facecolor=dark_color,
                    edgecolor="none",
                    alpha=0.95,
                )

        ax_mag.set_xlabel("Interval End Time (days)")
        ax_mag.set_ylabel("Mean |df/f per day|")
        ax_mag.set_title("Neighbor Pair Drift Magnitude vs Time")
        ax_mag.grid(True, alpha=0.3)
        if np.any(mag_mask):
            ax_mag.legend(loc="best", fontsize=8)

        self.res_neighbor_corr_figure.tight_layout()
        if self.res_neighbor_corr_status_var is not None:
            origin_dt = data.get("elapsed_time_origin")
            origin_text = origin_dt.strftime("%Y-%m-%d") if isinstance(origin_dt, datetime) else "unknown"
            ref_interval = summary["reference_interval"]
            self.res_neighbor_corr_status_var.set(
                f"Elapsed-time origin: {origin_text}. Showing {len(points)} consecutive interval(s) across "
                f"{len(data['pair_series'])} neighboring pair(s). Top: "
                f"{'colored traces show per-pair contribution, ' if show_curves else ''}"
                f"grey band = middle 50%, black line = mean correlation with the reference interval ending at "
                f"{float(ref_interval['end_elapsed_days']):.3f} day(s). Bottom: mean absolute pair drift rate with +/- 1 std; "
                f"threshold {threshold_rel:.4f} df/f."
            )
        self.res_neighbor_corr_canvas.draw_idle()
