from __future__ import annotations

from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from scipy import stats
from tkinter import messagebox

from analysis_gui_support.analysis_io import _dataset_dir
from analysis_gui_support.analysis_models import _make_event

class ResonatorNeighborDfrelWindowMixin:
    def open_resonator_neighbor_dfrel_window(self) -> None:
        if self.res_neighbor_dfrel_window is not None and self.res_neighbor_dfrel_window.winfo_exists():
            self.res_neighbor_dfrel_window.lift()
            self._render_resonator_neighbor_dfrel_window()
            return

        self.res_neighbor_dfrel_window = tk.Toplevel(self.root)
        self.res_neighbor_dfrel_window.title("Neighbor Pair df/f vs Time")
        self.res_neighbor_dfrel_window.geometry("1380x860")
        self.res_neighbor_dfrel_window.protocol("WM_DELETE_WINDOW", self._close_resonator_neighbor_dfrel_window)

        controls = tk.Frame(self.res_neighbor_dfrel_window, padx=8, pady=8)
        controls.pack(side="top", fill="x")
        control_row = tk.Frame(controls)
        control_row.pack(side="top", fill="x", anchor="w")
        self.res_neighbor_dfrel_status_var = tk.StringVar(
            value="Showing neighboring resonator-pair separation df/f versus elapsed time."
        )
        self.res_neighbor_dfrel_sep_rel_var = tk.DoubleVar(value=0.004)
        self.res_neighbor_dfrel_show_iqr_var = tk.BooleanVar(value=True)
        self.res_neighbor_dfrel_mode_var = tk.StringVar(value="drift")
        self.res_neighbor_dfrel_initial_date_var = tk.StringVar(value=self._dataset_res_neighbor_initial_date())

        tk.Label(control_row, text="Initial Date").pack(side="left", padx=(0, 4))
        initial_date_entry = tk.Entry(control_row, width=12, textvariable=self.res_neighbor_dfrel_initial_date_var)
        initial_date_entry.pack(side="left", padx=(0, 4))
        initial_date_entry.bind(
            "<Return>",
            lambda _event: (self._sync_res_neighbor_initial_date(autosave=True), self._render_resonator_neighbor_dfrel_window()),
        )
        initial_date_entry.bind(
            "<FocusOut>",
            lambda _event: (self._sync_res_neighbor_initial_date(autosave=True), self._render_resonator_neighbor_dfrel_window()),
        )
        tk.Label(control_row, text="YYYY-MM-DD").pack(side="left", padx=(0, 12))
        self.res_neighbor_dfrel_sep_scale = tk.Scale(
            control_row,
            from_=0.0,
            to=0.04,
            resolution=0.0001,
            orient="horizontal",
            length=260,
            digits=5,
            label="Max pair mean separation (df/f)",
            variable=self.res_neighbor_dfrel_sep_rel_var,
            command=lambda _value: self._render_resonator_neighbor_dfrel_window(),
        )
        self.res_neighbor_dfrel_sep_scale.pack(side="left", padx=(0, 12))
        tk.Radiobutton(
            control_row,
            text="Mean +/- Std Spacing",
            value="drift",
            variable=self.res_neighbor_dfrel_mode_var,
            command=self._render_resonator_neighbor_dfrel_window,
        ).pack(side="left", padx=(0, 8))
        tk.Radiobutton(
            control_row,
            text="Spacing Displacement",
            value="change",
            variable=self.res_neighbor_dfrel_mode_var,
            command=self._render_resonator_neighbor_dfrel_window,
        ).pack(side="left", padx=(0, 8))
        tk.Radiobutton(
            control_row,
            text="Spacing",
            value="spacing",
            variable=self.res_neighbor_dfrel_mode_var,
            command=self._render_resonator_neighbor_dfrel_window,
        ).pack(side="left", padx=(0, 8))
        tk.Checkbutton(
            control_row,
            text="Summary Band",
            variable=self.res_neighbor_dfrel_show_iqr_var,
            command=self._render_resonator_neighbor_dfrel_window,
        ).pack(side="left", padx=(0, 12))
        tk.Button(control_row, text="Show On Scans", width=13, command=self.open_resonator_neighbor_scan_window).pack(
            side="left",
            padx=(0, 8),
        )
        tk.Button(control_row, text="Refresh", width=10, command=self._render_resonator_neighbor_dfrel_window).pack(
            side="left",
            padx=(0, 8),
        )
        tk.Label(controls, textvariable=self.res_neighbor_dfrel_status_var, anchor="w", justify="left").pack(
            side="top",
            fill="x",
            expand=True,
            pady=(6, 0),
        )

        self.res_neighbor_dfrel_figure = Figure(figsize=(12.5, 7))
        self.res_neighbor_dfrel_canvas = FigureCanvasTkAgg(
            self.res_neighbor_dfrel_figure,
            master=self.res_neighbor_dfrel_window,
        )
        self.res_neighbor_dfrel_toolbar = NavigationToolbar2Tk(
            self.res_neighbor_dfrel_canvas,
            self.res_neighbor_dfrel_window,
        )
        self.res_neighbor_dfrel_toolbar.update()
        self.res_neighbor_dfrel_toolbar.pack(side="top", fill="x")
        self.res_neighbor_dfrel_canvas.get_tk_widget().pack(fill="both", expand=True)

        self._render_resonator_neighbor_dfrel_window()



    def _close_resonator_neighbor_dfrel_window(self) -> None:
        if self.res_neighbor_dfrel_window is not None and self.res_neighbor_dfrel_window.winfo_exists():
            self.res_neighbor_dfrel_window.destroy()
        self.res_neighbor_dfrel_window = None
        self.res_neighbor_dfrel_canvas = None
        self.res_neighbor_dfrel_toolbar = None
        self.res_neighbor_dfrel_figure = None
        self.res_neighbor_dfrel_status_var = None
        self.res_neighbor_dfrel_sep_rel_var = None
        self.res_neighbor_dfrel_show_iqr_var = None
        self.res_neighbor_dfrel_mode_var = None
        self.res_neighbor_dfrel_initial_date_var = None
        self.res_neighbor_dfrel_sep_scale = None
        self._res_neighbor_dfrel_ax = None



    def _render_resonator_neighbor_dfrel_window(self) -> None:
        if self.res_neighbor_dfrel_figure is None or self.res_neighbor_dfrel_canvas is None:
            return

        self.res_neighbor_dfrel_figure.clear()
        ax = self.res_neighbor_dfrel_figure.add_subplot(111)
        self._res_neighbor_dfrel_ax = ax

        threshold_rel = (
            float(self.res_neighbor_dfrel_sep_rel_var.get())
            if self.res_neighbor_dfrel_sep_rel_var is not None
            else 0.004
        )
        initial_date_text = (
            str(self.res_neighbor_dfrel_initial_date_var.get())
            if self.res_neighbor_dfrel_initial_date_var is not None
            else ""
        )
        try:
            overlay_state = self._resonator_neighbor_scan_overlay_state(
                threshold_rel,
                initial_date_text=initial_date_text,
            )
        except Exception as exc:
            ax.text(0.5, 0.5, str(exc), ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            if self.res_neighbor_dfrel_status_var is not None:
                self.res_neighbor_dfrel_status_var.set(str(exc))
            self.res_neighbor_dfrel_canvas.draw_idle()
            return

        data = overlay_state["data"]
        mode = (
            str(self.res_neighbor_dfrel_mode_var.get())
            if self.res_neighbor_dfrel_mode_var is not None
            else "change"
        ).strip().lower()
        pair_series = self._resonator_neighbor_plot_series(data["pair_series"], mode)
        mean_pair_freqs_hz = np.asarray(data["mean_pair_freqs_hz"], dtype=float)
        norm = overlay_state["norm"]
        cmap = overlay_state["cmap"]
        vmin = float(norm.vmin)
        vmax = float(norm.vmax)
        show_iqr = bool(self.res_neighbor_dfrel_show_iqr_var.get()) if self.res_neighbor_dfrel_show_iqr_var is not None else True

        summary = []
        drift_single_interval_xlim: Optional[tuple[float, float]] = None
        if mode == "drift":
            summary = self._resonator_neighbor_drift_rate_summary(pair_series)
            drift_series = self._resonator_neighbor_drift_rate_series(pair_series)
            if summary:
                x_summary = np.asarray([float(item["elapsed_days"]) for item in summary], dtype=float)
                lower_summary = np.asarray([float(item["lower"]) for item in summary], dtype=float)
                upper_summary = np.asarray([float(item["upper"]) for item in summary], dtype=float)
                if show_iqr:
                    valid_mask = np.isfinite(x_summary) & np.isfinite(lower_summary) & np.isfinite(upper_summary)
                    valid_count = int(np.count_nonzero(valid_mask))
                    if valid_count >= 2:
                        ax.fill_between(
                            x_summary[valid_mask],
                            lower_summary[valid_mask],
                            upper_summary[valid_mask],
                            color="0.15",
                            alpha=0.28,
                            zorder=3.2,
                            linewidth=0.0,
                            label="Mean +/- 1 std",
                        )
                        ax.plot(
                            x_summary[valid_mask],
                            lower_summary[valid_mask],
                            color="black",
                            linewidth=0.9,
                            alpha=0.9,
                            zorder=3.35,
                        )
                        ax.plot(
                            x_summary[valid_mask],
                            upper_summary[valid_mask],
                            color="black",
                            linewidth=0.9,
                            alpha=0.9,
                            zorder=3.35,
                        )
                    elif valid_count == 1:
                        idx = int(np.flatnonzero(valid_mask)[0])
                        x0 = float(x_summary[idx])
                        mean0 = float(summary[idx]["mean"])
                        lower0 = float(lower_summary[idx])
                        upper0 = float(upper_summary[idx])
                        x_left = x0 - 0.5
                        x_right = x0 + 0.5
                        ax.fill_between(
                            np.asarray([x_left, x_right], dtype=float),
                            np.asarray([lower0, lower0], dtype=float),
                            np.asarray([upper0, upper0], dtype=float),
                            color="0.15",
                            alpha=0.28,
                            zorder=3.2,
                            linewidth=0.0,
                            label="Mean +/- 1 std",
                        )
                        ax.plot(
                            np.asarray([x_left, x_right, x_right, x_left, x_left], dtype=float),
                            np.asarray([lower0, lower0, upper0, upper0, lower0], dtype=float),
                            color="black",
                            linewidth=1.0,
                            alpha=0.95,
                            zorder=3.35,
                        )
                        ax.plot(
                            np.asarray([x_left, x_right], dtype=float),
                            np.asarray([mean0, mean0], dtype=float),
                            color="black",
                            linewidth=1.6,
                            alpha=0.95,
                            zorder=4.3,
                        )
                        drift_single_interval_xlim = (x0 - 5.0, x0 + 5.0)
            for pair in drift_series:
                color = pair["color"]
                x = np.asarray([float(pt["elapsed_days"]) for pt in pair["drift_points"]], dtype=float)
                y = np.asarray([float(pt["drift_rate"]) for pt in pair["drift_points"]], dtype=float)
                ax.plot(
                    x,
                    y,
                    color=color,
                    linewidth=1.5,
                    alpha=0.9,
                    marker="o",
                    markersize=4.0,
                    zorder=4.0,
                )
        else:
            summary = self._resonator_neighbor_summary_by_time(pair_series)
            if show_iqr and summary:
                x_summary = np.asarray([float(item["elapsed_days"]) for item in summary], dtype=float)
                lower = np.asarray([float(item["lower"]) for item in summary], dtype=float)
                upper = np.asarray([float(item["upper"]) for item in summary], dtype=float)
                q1 = np.asarray([float(item["q1"]) for item in summary], dtype=float)
                median = np.asarray([float(item["median"]) for item in summary], dtype=float)
                q3 = np.asarray([float(item["q3"]) for item in summary], dtype=float)
                ax.fill_between(
                    x_summary,
                    lower,
                    upper,
                    color="0.85",
                    alpha=0.6,
                    zorder=1.4,
                    linewidth=0.0,
                    label="Mean +/- 1 std",
                )
                ax.fill_between(
                    x_summary,
                    q1,
                    q3,
                    color="0.15",
                    alpha=0.42,
                    zorder=3.2,
                    linewidth=0.0,
                    label="Middle 50%",
                )
                ax.plot(
                    x_summary,
                    q1,
                    color="black",
                    linewidth=0.9,
                    alpha=0.95,
                    zorder=3.35,
                )
                ax.plot(
                    x_summary,
                    q3,
                    color="black",
                    linewidth=0.9,
                    alpha=0.95,
                    zorder=3.35,
                )
                ax.plot(
                    x_summary,
                    median,
                    color="black",
                    linewidth=6.0,
                    alpha=1.0,
                    zorder=4.2,
                    label="Median",
                )

            for pair in pair_series:
                color = pair["color"]
                x = np.asarray([float(pt["elapsed_days"]) for pt in pair["points"]], dtype=float)
                y = np.asarray([float(pt.get("plot_df_over_f", pt["df_over_f"])) for pt in pair["points"]], dtype=float)
                ax.plot(
                    x,
                    y,
                    color=color,
                    linewidth=1.5,
                    alpha=0.9,
                    marker="o",
                    markersize=4.0,
                )

        ax.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Elapsed Time (days)")
        if mode == "drift":
            ax.set_ylabel("Neighbor Pair Gap Drift Rate (df/f per day)")
            ax.set_title("Neighbor Pair Gap Drift Rate vs Time")
        elif mode == "change":
            ax.set_ylabel("Neighbor Pair Separation Displacement df/f")
            ax.set_title("Neighboring Resonator-Pair Separation Displacement vs Time")
        else:
            ax.set_ylabel("Neighbor Pair Separation df/f")
            ax.set_title("Neighboring Resonator-Pair Relative Separation vs Time")
        if mode == "drift" and drift_single_interval_xlim is not None:
            ax.set_xlim(*drift_single_interval_xlim)

        if show_iqr and summary:
            ax.legend(loc="best", fontsize=8)

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        colorbar = self.res_neighbor_dfrel_figure.colorbar(sm, ax=ax, pad=0.02)
        colorbar.set_label("Pair Mean Frequency (GHz)")
        tick_vals = colorbar.get_ticks()
        tick_vals = [tick for tick in tick_vals if vmin <= float(tick) <= vmax]
        if tick_vals:
            colorbar.set_ticks(tick_vals)
        colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value / 1.0e9:.3f}"))

        freq_span_hz = max(vmax - vmin, 1.0)
        sorted_pair_freqs_hz = np.sort(mean_pair_freqs_hz)
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
            colorbar.ax.axhspan(lo_hz, hi_hz, xmin=0.0, xmax=1.0, facecolor=dark_color, edgecolor="none", alpha=0.95)

        self.res_neighbor_dfrel_figure.tight_layout()
        if self.res_neighbor_dfrel_status_var is not None:
            origin_dt = data.get("elapsed_time_origin")
            origin_text = (
                origin_dt.strftime("%Y-%m-%d")
                if isinstance(origin_dt, datetime)
                else "unknown"
            )
            origin_prefix = f"Elapsed-time origin: {origin_text}. "
            if mode == "drift":
                summary_text = " Summary overlay: grey band = mean +/- 1 std; colored traces = individual pair drift rates." if show_iqr else ""
                self.res_neighbor_dfrel_status_var.set(
                    f"{origin_prefix}Showing mean/std of pair-gap drift rate across {len(summary)} elapsed-time point(s) from {len(pair_series)} neighboring pair(s) and {len(data['tests'])} selected test unit(s); threshold {threshold_rel:.4f} df/f.{summary_text}"
                )
            else:
                summary_text = " Summary overlay: grey band = middle 50%, black line = median." if show_iqr else ""
                mode_text = "spacing displacement from initial" if mode == "change" else "spacing"
                self.res_neighbor_dfrel_status_var.set(
                    f"{origin_prefix}Showing {len(pair_series)} neighboring pair curve(s) across {len(data['tests'])} selected test unit(s) in {mode_text} mode; threshold {threshold_rel:.4f} df/f.{summary_text}"
                )
        self.res_neighbor_dfrel_canvas.draw_idle()
        if self.res_neighbor_scan_window is not None and self.res_neighbor_scan_window.winfo_exists():
            self._render_resonator_neighbor_scan_window()



    def _draw_resonator_neighbor_scan_overlay(
        self,
        ax,
        *,
        rows: list[dict],
        overlay_state: dict,
        xlim_ghz: Optional[tuple[float, float]] = None,
        spacing: float = 1.5,
        truncate_threshold: float = 1.5,
        title: str = "",
    ) -> dict:
        data = overlay_state["data"]
        pair_series = data["pair_series"]
        resonator_colors = overlay_state["resonator_colors"]
        neutral_color = (0.45, 0.45, 0.45, 1.0)
        offset_by_scan_key, tick_info = self._attached_resonance_editor_offset_map(rows, spacing)

        freq_min = min(float(row["freq"][0]) for row in rows) / 1.0e9
        freq_max = max(float(row["freq"][-1]) for row in rows) / 1.0e9
        x_pad = max((freq_max - freq_min) * 0.01, 1e-6)
        resonator_tracks: dict[str, list[tuple[float, float]]] = {}
        points_by_scan_key: dict[str, dict[str, dict]] = {}
        total_markers = 0

        for row in rows:
            scan_key = str(row["scan_key"])
            offset = float(offset_by_scan_key[scan_key])
            freq_ghz = np.asarray(row["freq"], dtype=float) / 1.0e9
            amp_display = np.minimum(np.asarray(row["amp"], dtype=float), truncate_threshold)
            y = amp_display + offset
            ax.plot(freq_ghz, y, linewidth=1.0, color="tab:blue", alpha=0.8, zorder=1)

            row_points: dict[str, dict] = {}
            for resonator in row["resonators"]:
                resonator_label = str(resonator["resonator_number"])
                target_hz = float(resonator["target_hz"])
                target_ghz = target_hz / 1.0e9
                if xlim_ghz is not None and not (xlim_ghz[0] <= target_ghz <= xlim_ghz[1]):
                    continue
                y_pt = self._interpolate_y(row["freq"], amp_display, target_hz) + offset
                color = resonator_colors.get(resonator_label, neutral_color)
                row_points[resonator_label] = {"x_ghz": target_ghz, "y": y_pt, "color": color}
                resonator_tracks.setdefault(resonator_label, []).append((target_ghz, y_pt))
                ax.plot(
                    [target_ghz],
                    [y_pt],
                    linestyle="none",
                    marker="o",
                    markersize=6,
                    markerfacecolor="none",
                    markeredgecolor=color,
                    markeredgewidth=1.5,
                    zorder=4,
                )
                ax.text(
                    target_ghz,
                    y_pt - 0.18,
                    resonator_label,
                    ha="center",
                    va="top",
                    fontsize=8,
                    color=color,
                    zorder=5,
                )
                total_markers += 1
            points_by_scan_key[scan_key] = row_points

        for resonator_label, points in resonator_tracks.items():
            if len(points) < 2:
                continue
            points = sorted(points, key=lambda item: item[1], reverse=True)
            ax.plot(
                [pt[0] for pt in points],
                [pt[1] for pt in points],
                linestyle=":",
                linewidth=2.0,
                color=resonator_colors.get(resonator_label, neutral_color),
                alpha=0.95,
                zorder=2,
            )

        connector_count = 0
        visible_pair_labels: set[str] = set()
        for row in rows:
            row_points = points_by_scan_key.get(str(row["scan_key"]), {})
            for pair in pair_series:
                low_label = str(pair["low_label"])
                high_label = str(pair["high_label"])
                if low_label not in row_points or high_label not in row_points:
                    continue
                low_pt = row_points[low_label]
                high_pt = row_points[high_label]
                ax.plot(
                    [float(low_pt["x_ghz"]), float(high_pt["x_ghz"])],
                    [float(low_pt["y"]), float(high_pt["y"])],
                    linestyle=":",
                    linewidth=2.4,
                    color=pair["color"],
                    alpha=0.9,
                    zorder=3,
                )
                connector_count += 1
                visible_pair_labels.add(str(pair["label"]))

        y_low = min(
            float(np.min(np.minimum(np.asarray(row["amp"], dtype=float), truncate_threshold)))
            + float(offset_by_scan_key[str(row["scan_key"])])
            for row in rows
        )
        y_high = max(
            float(np.max(np.minimum(np.asarray(row["amp"], dtype=float), truncate_threshold)))
            + float(offset_by_scan_key[str(row["scan_key"])])
            for row in rows
        )
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Normalized |S21| + vertical offset")
        ax.grid(True, alpha=0.3)
        ax.set_yticks([item[0] for item in tick_info])
        ax.set_yticklabels([item[1] for item in tick_info], fontsize=8)
        ax.set_ylim(y_low - 0.2, y_high + 0.2)
        if xlim_ghz is None:
            ax.set_xlim(freq_min - 0.5 * x_pad, freq_max + 2.0 * x_pad)
        else:
            ax.set_xlim(xlim_ghz)
        if title:
            ax.set_title(title)
        return {
            "marker_count": total_markers,
            "connector_count": connector_count,
            "visible_pair_labels": visible_pair_labels,
        }



    def open_resonator_neighbor_scan_window(self, source: str = "dfrel") -> None:
        self._res_neighbor_scan_source = str(source or "dfrel").strip().lower()
        if self.res_neighbor_scan_window is not None and self.res_neighbor_scan_window.winfo_exists():
            self.res_neighbor_scan_window.lift()
            self._render_resonator_neighbor_scan_window()
            return

        self.res_neighbor_scan_window = tk.Toplevel(self.root)
        self.res_neighbor_scan_window.title("Neighbor Pair Resonators On Scans")
        self.res_neighbor_scan_window.geometry("1380x900")
        self.res_neighbor_scan_window.protocol("WM_DELETE_WINDOW", self._close_resonator_neighbor_scan_window)

        controls = tk.Frame(self.res_neighbor_scan_window, padx=8, pady=8)
        controls.pack(side="top", fill="x")
        self.res_neighbor_scan_status_var = tk.StringVar(
            value="Showing marked resonators on selected VNA scans with active neighboring-pair coloring."
        )
        tk.Label(controls, textvariable=self.res_neighbor_scan_status_var, anchor="w", justify="left").pack(
            side="left",
            fill="x",
            expand=True,
        )
        tk.Button(controls, text="Save PDF", width=10, command=self._save_resonator_neighbor_scan_pdf).pack(
            side="right",
            padx=(0, 8),
        )
        tk.Button(controls, text="Refresh", width=10, command=self._render_resonator_neighbor_scan_window).pack(
            side="right"
        )

        self.res_neighbor_scan_figure = Figure(figsize=(12.5, 7.5))
        self.res_neighbor_scan_canvas = FigureCanvasTkAgg(
            self.res_neighbor_scan_figure,
            master=self.res_neighbor_scan_window,
        )
        self.res_neighbor_scan_toolbar = NavigationToolbar2Tk(
            self.res_neighbor_scan_canvas,
            self.res_neighbor_scan_window,
        )
        self.res_neighbor_scan_toolbar.update()
        self.res_neighbor_scan_toolbar.pack(side="top", fill="x")
        self.res_neighbor_scan_canvas.get_tk_widget().pack(fill="both", expand=True)

        self._render_resonator_neighbor_scan_window()



    def _close_resonator_neighbor_scan_window(self) -> None:
        if self.res_neighbor_scan_window is not None and self.res_neighbor_scan_window.winfo_exists():
            self.res_neighbor_scan_window.destroy()
        self.res_neighbor_scan_window = None
        self.res_neighbor_scan_canvas = None
        self.res_neighbor_scan_toolbar = None
        self.res_neighbor_scan_figure = None
        self.res_neighbor_scan_status_var = None
        self._res_neighbor_scan_source = "dfrel"
        self._res_neighbor_scan_ax = None



    def _render_resonator_neighbor_scan_window(self) -> None:
        if self.res_neighbor_scan_figure is None or self.res_neighbor_scan_canvas is None:
            return

        self.res_neighbor_scan_figure.clear()
        ax = self.res_neighbor_scan_figure.add_subplot(111)
        self._res_neighbor_scan_ax = ax

        threshold_rel, initial_date_text = self._resonator_neighbor_scan_control_values()
        rows, warnings = self._selected_scans_for_attached_resonance_editor()
        if not rows:
            message = "No selected scans with normalized data."
            if warnings:
                message += " Missing normalized data for: " + ", ".join(warnings[:6])
            ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            if self.res_neighbor_scan_status_var is not None:
                self.res_neighbor_scan_status_var.set(message)
            self.res_neighbor_scan_canvas.draw_idle()
            return

        try:
            overlay_state = self._resonator_neighbor_scan_overlay_state(
                threshold_rel,
                initial_date_text=initial_date_text,
            )
        except Exception as exc:
            ax.text(0.5, 0.5, str(exc), ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            if self.res_neighbor_scan_status_var is not None:
                self.res_neighbor_scan_status_var.set(str(exc))
            self.res_neighbor_scan_canvas.draw_idle()
            return

        data = overlay_state["data"]
        draw_stats = self._draw_resonator_neighbor_scan_overlay(
            ax,
            rows=rows,
            overlay_state=overlay_state,
            title="Marked Resonators On Selected Scans With Neighbor-Pair Coloring",
        )

        self.res_neighbor_scan_figure.tight_layout()
        if self.res_neighbor_scan_status_var is not None:
            status = (
                f"Showing {int(draw_stats['marker_count'])} marker(s), {len(data['pair_series'])} active neighboring pair(s), and "
                f"{int(draw_stats['connector_count'])} same-scan pair connector(s); threshold {threshold_rel:.4f} df/f."
            )
            if warnings:
                status += " Missing normalized data for: " + ", ".join(warnings[:6])
                if len(warnings) > 6:
                    status += f", ... (+{len(warnings) - 6} more)"
            self.res_neighbor_scan_status_var.set(status)
        self.res_neighbor_scan_canvas.draw_idle()



    def _save_resonator_neighbor_scan_pdf(self) -> None:
        threshold_rel, initial_date_text = self._resonator_neighbor_scan_control_values()
        rows, warnings = self._selected_scans_for_attached_resonance_editor()
        if not rows:
            messagebox.showwarning("No plottable scans", "No selected scans with normalized data are available.")
            return

        try:
            overlay_state = self._resonator_neighbor_scan_overlay_state(
                threshold_rel,
                initial_date_text=initial_date_text,
            )
        except Exception as exc:
            messagebox.showerror("Cannot save PDF", str(exc), parent=self.res_neighbor_scan_window)
            return

        pair_series = overlay_state["data"]["pair_series"]
        pair_mid_freqs_hz = sorted(float(pair["mean_freq_hz"]) for pair in pair_series)
        if not pair_mid_freqs_hz:
            messagebox.showwarning("No active pairs", "No active neighboring pairs are available for PDF export.")
            return

        out_dir = _dataset_dir(self.dataset) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_neighbor_pair_scan_plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = out_dir / "neighbor_pair_scan_overlay.pdf"

        freq_min_hz = min(float(row["freq"][0]) for row in rows)
        freq_max_hz = max(float(row["freq"][-1]) for row in rows)
        zoom_span_hz = 100.0e6
        zoom_step_hz = 80.0e6
        zoom_windows: list[tuple[float, float]] = []
        if freq_max_hz - freq_min_hz > zoom_span_hz:
            start_hz = freq_min_hz
            seen_starts: set[float] = set()
            while True:
                rounded_start = round(start_hz, 3)
                if rounded_start in seen_starts:
                    break
                seen_starts.add(rounded_start)
                end_hz = min(start_hz + zoom_span_hz, freq_max_hz)
                if any(start_hz <= pair_hz <= end_hz for pair_hz in pair_mid_freqs_hz):
                    zoom_windows.append((start_hz, end_hz))
                if end_hz >= freq_max_hz:
                    break
                next_start = start_hz + zoom_step_hz
                if next_start + zoom_span_hz > freq_max_hz:
                    next_start = max(freq_min_hz, freq_max_hz - zoom_span_hz)
                if next_start <= start_hz:
                    break
                start_hz = next_start

        page_count = 0
        marker_count = 0
        connector_count = 0
        with PdfPages(pdf_path) as pdf:
            fig = Figure(figsize=(12, 7))
            ax = fig.add_subplot(111)
            stats = self._draw_resonator_neighbor_scan_overlay(
                ax,
                rows=rows,
                overlay_state=overlay_state,
                title="Neighbor Pair Scan Overlay | Full Span",
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            page_count += 1
            marker_count += int(stats["marker_count"])
            connector_count += int(stats["connector_count"])

            for page_idx, (lo_hz, hi_hz) in enumerate(zoom_windows, start=1):
                fig = Figure(figsize=(12, 7))
                ax = fig.add_subplot(111)
                stats = self._draw_resonator_neighbor_scan_overlay(
                    ax,
                    rows=rows,
                    overlay_state=overlay_state,
                    xlim_ghz=(lo_hz / 1.0e9, hi_hz / 1.0e9),
                    title=(
                        f"Neighbor Pair Scan Overlay | Zoom {page_idx} | "
                        f"{lo_hz / 1.0e9:.6f} to {hi_hz / 1.0e9:.6f} GHz"
                    ),
                )
                if not stats["visible_pair_labels"]:
                    plt.close(fig)
                    continue
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
                page_count += 1
                marker_count += int(stats["marker_count"])
                connector_count += int(stats["connector_count"])

        self.dataset.processing_history.append(
            _make_event(
                "save_neighbor_pair_scan_overlay_pdf",
                {
                    "output_pdf": str(pdf_path),
                    "page_count": int(page_count),
                    "marker_count": int(marker_count),
                    "connector_count": int(connector_count),
                    "threshold_rel": float(threshold_rel),
                    "zoom_span_mhz": 100.0,
                    "zoom_overlap_mhz": 20.0,
                },
            )
        )
        self._mark_dirty()
        self._autosave_dataset()
        for warning in warnings[:20]:
            self._log(f"Neighbor pair scan overlay warning: {warning}")
        self._log(f"Saved neighbor pair scan overlay PDF: {pdf_path}")
        if self.res_neighbor_scan_status_var is not None:
            self.res_neighbor_scan_status_var.set(
                f"Saved overlay PDF with {page_count} page(s) to {pdf_path}"
            )
