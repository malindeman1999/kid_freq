from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from tkinter import messagebox

from ..analysis_io import _dataset_dir
from ..analysis_models import _make_event, _read_polar_series


class AnalysisSelectorPlotMixin:
    def open_analysis_selector(self) -> None:
        if not self.dataset.vna_scans:
            messagebox.showwarning("No data", "No VNA scans are loaded in this dataset.")
            return

        selector = tk.Toplevel(self.root)
        selector.title("Select VNA Scans for Analysis")
        selector.geometry("760x360")

        tk.Label(selector, text="Choose scan(s) for analysis:").pack(
            anchor="w", padx=10, pady=(10, 4)
        )

        listbox = tk.Listbox(selector, width=120, height=14, selectmode=tk.MULTIPLE)
        listbox.pack(fill="both", expand=True, padx=10, pady=4)

        selected_keys = set(self.dataset.selected_scan_keys)
        for idx, scan in enumerate(self.dataset.vna_scans):
            file_timestamp = str(getattr(scan, "file_timestamp", "")).strip() or "unknown"
            group_text = f" | group {int(scan.plot_group)}" if scan.plot_group is not None else ""
            label = (
                f"{idx:03d} | {Path(scan.filename).name} | "
                f"file {file_timestamp} | loaded {scan.loaded_at}{group_text}"
            )
            listbox.insert(tk.END, label)
            if self._scan_key(scan) in selected_keys:
                listbox.selection_set(idx)

        def apply_selection() -> None:
            indices = listbox.curselection()
            self.dataset.selected_scan_keys = [
                self._scan_key(self.dataset.vna_scans[i]) for i in indices
            ]
            self._mark_dirty()
            self._refresh_status()
            self._log(f"Analysis scan selection updated: {len(indices)} selected.")
            self._autosave_dataset()
            selector.destroy()

        button_frame = tk.Frame(selector)
        button_frame.pack(fill="x", padx=10, pady=(2, 10))
        tk.Button(button_frame, text="Select All", command=lambda: listbox.select_set(0, tk.END)).pack(
            side="left"
        )
        tk.Button(button_frame, text="Clear All", command=lambda: listbox.selection_clear(0, tk.END)).pack(
            side="left", padx=(6, 0)
        )
        tk.Button(button_frame, text="Apply Selection", command=apply_selection).pack(side="right")

    def plot_selected_vna_scans(self) -> None:
        if not self.dataset.vna_scans:
            messagebox.showwarning("No data", "No VNA scans are loaded in this dataset.")
            return

        scans = self._selected_scans()
        if not scans:
            messagebox.showwarning(
                "No selection",
                "No scans selected for analysis.\nUse 'Select Scans for Analysis' first.",
            )
            return

        if self.plot_scans_window is not None and self.plot_scans_window.winfo_exists():
            self.plot_scans_window.lift()
            self._plot_scans_render()
            return

        self.plot_scans_window = tk.Toplevel(self.root)
        self.plot_scans_window.title("Selected VNA Scans")
        self.plot_scans_window.geometry("1280x900")
        self.plot_scans_window.protocol("WM_DELETE_WINDOW", self._plot_scans_close)

        controls = tk.Frame(self.plot_scans_window, padx=8, pady=8)
        controls.pack(side="top", fill="x")
        self.plot_scans_show_amp_var = tk.BooleanVar(value=True)
        self.plot_scans_show_phase_var = tk.BooleanVar(value=False)
        self.plot_scans_group_var = tk.BooleanVar(value=True)
        self.plot_scans_data_mode_var = tk.StringVar(value="raw")
        self.plot_scans_show_gaussian_var = tk.BooleanVar(value=False)
        self.plot_scans_show_dsdf_var = tk.BooleanVar(value=False)
        self.plot_scans_show_2pi_var = tk.BooleanVar(value=False)
        self.plot_scans_show_vna_phase_var = tk.BooleanVar(value=False)
        self.plot_scans_show_other_phase_var = tk.BooleanVar(value=False)
        self.plot_scans_show_attached_res_var = tk.BooleanVar(value=True)
        self.plot_scans_auto_y_var = tk.BooleanVar(value=True)
        self.plot_scans_raw_radio = tk.Radiobutton(
            controls,
            text="Raw VNA data",
            variable=self.plot_scans_data_mode_var,
            value="raw",
            command=self._plot_scans_on_toggle_changed,
        )
        self.plot_scans_raw_radio.pack(side="left", padx=(0, 12))
        self.plot_scans_normalized_radio = tk.Radiobutton(
            controls,
            text="Baseline normalized data",
            variable=self.plot_scans_data_mode_var,
            value="normalized",
            command=self._plot_scans_on_toggle_changed,
        )
        self.plot_scans_normalized_radio.pack(side="left", padx=(0, 12))
        tk.Checkbutton(
            controls,
            text="Show amplitude",
            variable=self.plot_scans_show_amp_var,
            command=self._plot_scans_on_toggle_changed,
        ).pack(side="left", padx=(0, 12))
        tk.Checkbutton(
            controls,
            text="Show phase",
            variable=self.plot_scans_show_phase_var,
            command=self._plot_scans_on_toggle_changed,
        ).pack(side="left", padx=(0, 12))
        tk.Checkbutton(
            controls,
            text="Combine selected scans by group",
            variable=self.plot_scans_group_var,
            command=self._plot_scans_on_toggle_changed,
        ).pack(side="left", padx=(0, 12))
        tk.Checkbutton(
            controls,
            text="Auto-scale Y over visible X",
            variable=self.plot_scans_auto_y_var,
            command=self._plot_scans_on_auto_y_toggled,
        ).pack(side="left", padx=(0, 12))
        self.plot_scans_status_var = tk.StringVar(value="Loaded selected scans.")
        tk.Label(controls, textvariable=self.plot_scans_status_var, anchor="w").pack(
            side="left", fill="x", expand=True
        )

        marker_controls = tk.Frame(self.plot_scans_window, padx=8, pady=2)
        marker_controls.pack(side="top", fill="x")
        tk.Checkbutton(
            marker_controls,
            text="Attached resonators",
            variable=self.plot_scans_show_attached_res_var,
            command=self._plot_scans_on_toggle_changed,
        ).pack(side="left", padx=(0, 12))
        tk.Checkbutton(
            marker_controls,
            text="Gaussian candidates",
            variable=self.plot_scans_show_gaussian_var,
            command=self._plot_scans_on_toggle_changed,
        ).pack(side="left", padx=(0, 12))
        tk.Checkbutton(
            marker_controls,
            text="dS21/df peaks",
            variable=self.plot_scans_show_dsdf_var,
            command=self._plot_scans_on_toggle_changed,
        ).pack(side="left", padx=(0, 12))
        tk.Checkbutton(
            marker_controls,
            text="2pi phase corrections",
            variable=self.plot_scans_show_2pi_var,
            command=self._plot_scans_on_toggle_changed,
        ).pack(side="left", padx=(0, 12))
        tk.Checkbutton(
            marker_controls,
            text="VNA phase corrections",
            variable=self.plot_scans_show_vna_phase_var,
            command=self._plot_scans_on_toggle_changed,
        ).pack(side="left", padx=(0, 12))
        tk.Checkbutton(
            marker_controls,
            text="Other phase discontinuities",
            variable=self.plot_scans_show_other_phase_var,
            command=self._plot_scans_on_toggle_changed,
        ).pack(side="left", padx=(0, 12))

        actions = tk.Frame(self.plot_scans_window, padx=8, pady=6)
        actions.pack(side="top", fill="x")
        tk.Button(actions, text="Close", width=12, command=self._plot_scans_close).pack(side="right")
        tk.Button(actions, text="Reset View", width=12, command=self._plot_scans_reset_view).pack(
            side="right", padx=(8, 0)
        )
        tk.Button(actions, text="Save PDF", width=12, command=self._plot_scans_save_pdf).pack(
            side="right", padx=(8, 0)
        )

        self.plot_scans_figure = Figure(figsize=(12, 7))
        self.plot_scans_canvas = FigureCanvasTkAgg(self.plot_scans_figure, master=self.plot_scans_window)
        self.plot_scans_toolbar = NavigationToolbar2Tk(self.plot_scans_canvas, self.plot_scans_window)
        self.plot_scans_toolbar.update()
        self.plot_scans_toolbar.pack(side="top", fill="x")
        self.plot_scans_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.plot_scans_canvas.mpl_connect("button_release_event", lambda _e: self._plot_scans_autoscale_all_y())

        self._plot_scans_missing_normalized_warned = None
        self._plot_scans_update_data_mode_controls()
        self._plot_scans_render()

    def _plot_scans_visible_columns(self) -> list[str]:
        columns: list[str] = []
        if self.plot_scans_show_amp_var is not None and bool(self.plot_scans_show_amp_var.get()):
            columns.append("amp")
        if self.plot_scans_show_phase_var is not None and bool(self.plot_scans_show_phase_var.get()):
            columns.append("phase")
        return columns

    def _plot_scans_on_toggle_changed(self) -> None:
        columns = self._plot_scans_visible_columns()
        if not columns:
            if self.plot_scans_show_amp_var is not None:
                self.plot_scans_show_amp_var.set(True)
            columns = self._plot_scans_visible_columns()
            messagebox.showwarning(
                "No plot type selected",
                "At least one of amplitude or phase must be shown.",
            )
        self._plot_scans_update_data_mode_controls()
        self._plot_scans_render()

    def _plot_scans_on_auto_y_toggled(self) -> None:
        if self.plot_scans_auto_y_var is not None and bool(self.plot_scans_auto_y_var.get()):
            self._plot_scans_autoscale_all_y()
        elif self.plot_scans_status_var is not None:
            self.plot_scans_status_var.set("Auto Y disabled. Manual Y zoom enabled.")

    def _plot_scans_autoscale_y_for_visible_x(self, ax) -> None:
        if self.plot_scans_auto_y_var is None or not bool(self.plot_scans_auto_y_var.get()):
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

    def _plot_scans_autoscale_all_y(self) -> None:
        if self.plot_scans_figure is None or self.plot_scans_canvas is None:
            return
        if self.plot_scans_auto_y_var is None or not bool(self.plot_scans_auto_y_var.get()):
            return
        for ax in self.plot_scans_figure.axes:
            self._plot_scans_autoscale_y_for_visible_x(ax)
        self.plot_scans_canvas.draw_idle()
        if self.plot_scans_status_var is not None:
            self.plot_scans_status_var.set("Auto Y applied to visible frequency range.")

    def _plot_scans_global_freq_range(self, scans) -> tuple[float, float]:
        global_freq_min_ghz = min(np.min(np.asarray(scan.freq, dtype=float)) for scan in scans) / 1.0e9
        global_freq_max_ghz = max(np.max(np.asarray(scan.freq, dtype=float)) for scan in scans) / 1.0e9
        if global_freq_max_ghz <= global_freq_min_ghz:
            global_freq_max_ghz = global_freq_min_ghz + 1e-9
        return float(global_freq_min_ghz), float(global_freq_max_ghz)

    def _plot_scans_has_normalized_data(self, scan) -> bool:
        norm = scan.baseline_filter.get("normalized", {})
        if not isinstance(norm, dict):
            return False
        amp, phase = _read_polar_series(
            norm,
            amplitude_key="norm_amp",
            phase_key="norm_phase_deg_unwrapped",
        )
        return amp.shape == scan.freq.shape and phase.shape == scan.freq.shape

    def _plot_scans_update_data_mode_controls(self) -> None:
        scans = self._selected_scans()
        has_any_normalized = any(self._plot_scans_has_normalized_data(scan) for scan in scans)
        if self.plot_scans_normalized_radio is not None:
            self.plot_scans_normalized_radio.configure(state="normal" if has_any_normalized else "disabled")
        if not has_any_normalized and self.plot_scans_data_mode_var is not None:
            self.plot_scans_data_mode_var.set("raw")
        if self.plot_scans_data_mode_var is not None and self.plot_scans_data_mode_var.get() == "raw":
            self._plot_scans_missing_normalized_warned = None

    def _plot_scans_data_mode(self) -> str:
        if self.plot_scans_data_mode_var is None:
            return "raw"
        return str(self.plot_scans_data_mode_var.get())

    def _plot_scans_series_for_scan(self, scan):
        freq_ghz = np.asarray(scan.freq, dtype=float) / 1.0e9
        mode = self._plot_scans_data_mode()
        if mode == "normalized":
            norm = scan.baseline_filter.get("normalized", {})
            if not isinstance(norm, dict):
                return None
            amp, phase = _read_polar_series(
                norm,
                amplitude_key="norm_amp",
                phase_key="norm_phase_deg_unwrapped",
            )
            if amp.shape != scan.freq.shape or phase.shape != scan.freq.shape:
                return None
            return {
                "freq_ghz": freq_ghz,
                "amp": np.asarray(amp, dtype=float),
                "phase": np.asarray(phase, dtype=float),
                "phase_label": "Normalized Phase (deg)",
                "amp_label": "Normalized |S21|",
            }

        if scan.has_dewrapped_phase():
            phase = scan.phase_deg_unwrapped()
            phase_label = "Phase (dewrapped, deg)"
        else:
            phase = scan.phase_deg_wrapped_raw()
            phase_label = "Phase (raw wrapped, deg)"
        return {
            "freq_ghz": freq_ghz,
            "amp": scan.amplitude(),
            "phase": np.asarray(phase, dtype=float),
            "phase_label": phase_label,
            "amp_label": "|S21|",
        }

    def _plot_scans_nearest_values(self, query_freq_hz: np.ndarray, ref_freq_hz: np.ndarray, ref_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

    def _plot_scans_overlay_points(self, scan) -> dict[str, np.ndarray]:
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

    def _plot_scans_add_overlays(self, ax, scan, payload: dict, *, use_phase: bool, multi_scan: bool, used_overlay_labels: set[str]) -> None:
        ref_freq_hz = np.asarray(scan.freq, dtype=float)
        ref_y = np.asarray(payload["phase" if use_phase else "amp"], dtype=float)
        marker_defs = [
            (
                self.plot_scans_show_gaussian_var is not None and bool(self.plot_scans_show_gaussian_var.get()),
                "gaussian",
                dict(linestyle="none", marker="o", markersize=8, markerfacecolor="none", markeredgewidth=1.5, color="green"),
                "Gaussian candidates",
            ),
            (
                self.plot_scans_show_dsdf_var is not None and bool(self.plot_scans_show_dsdf_var.get()),
                "dsdf",
                dict(linestyle="none", marker="D", markersize=6, color="red"),
                "dS21/df peaks",
            ),
            (
                self.plot_scans_show_2pi_var is not None and bool(self.plot_scans_show_2pi_var.get()),
                "regular",
                dict(linestyle="none", marker="o", markersize=4, color="black"),
                "2pi phase corrections",
            ),
            (
                self.plot_scans_show_vna_phase_var is not None and bool(self.plot_scans_show_vna_phase_var.get()),
                "congruent",
                dict(linestyle="none", marker="o", markersize=5, color="pink"),
                "VNA phase corrections",
            ),
            (
                self.plot_scans_show_other_phase_var is not None and bool(self.plot_scans_show_other_phase_var.get()),
                "noncongruent",
                dict(linestyle="none", marker="o", markersize=5, color="blue"),
                "Other phase discontinuities",
            ),
        ]
        points = self._plot_scans_overlay_points(scan)
        for enabled, key, style, label in marker_defs:
            if not enabled:
                continue
            freq_pts = points[key]
            if freq_pts.size == 0:
                continue
            x_hz, y_pts = self._plot_scans_nearest_values(freq_pts, ref_freq_hz, ref_y)
            if x_hz.size == 0:
                continue
            plot_label = label if label not in used_overlay_labels else None
            ax.plot(
                x_hz / 1.0e9,
                y_pts,
                label=plot_label,
                **style,
            )
            if plot_label is not None:
                used_overlay_labels.add(label)

    def _plot_scans_attached_resonator_points(self, scan, payload: dict, *, use_phase: bool) -> list[dict]:
        attached = scan.candidate_resonators.get("sheet_resonances", {})
        assignments = attached.get("assignments") if isinstance(attached, dict) else {}
        if not isinstance(assignments, dict):
            return []
        ref_freq_hz = np.asarray(scan.freq, dtype=float)
        ref_y = np.asarray(payload["phase" if use_phase else "amp"], dtype=float)
        if ref_freq_hz.size == 0 or ref_y.size == 0 or ref_freq_hz.shape != ref_y.shape:
            return []
        points: list[dict] = []
        for resonator_number, record in assignments.items():
            if not isinstance(record, dict):
                continue
            try:
                target_hz = float(record.get("frequency_hz"))
            except Exception:
                continue
            x_hz, y_vals = self._plot_scans_nearest_values(
                np.asarray([target_hz], dtype=float),
                ref_freq_hz,
                ref_y,
            )
            if x_hz.size == 0 or y_vals.size == 0:
                continue
            points.append(
                {
                    "resonator_number": str(resonator_number).strip(),
                    "x_ghz": float(x_hz[0]) / 1.0e9,
                    "y": float(y_vals[0]),
                }
            )
        return points

    def _plot_scans_add_attached_resonator_overlays(self, ax, panel_payloads, *, use_phase: bool) -> None:
        if self.plot_scans_show_attached_res_var is None or not bool(self.plot_scans_show_attached_res_var.get()):
            return
        all_points: list[dict] = []
        for scan, payload in panel_payloads:
            points = self._plot_scans_attached_resonator_points(scan, payload, use_phase=use_phase)
            for point in points:
                all_points.append(point)

        y_text_offset = 0.02
        if all_points:
            y_all = np.asarray([point["y"] for point in all_points], dtype=float)
            y_span = float(np.max(y_all) - np.min(y_all)) if y_all.size else 0.0
            y_text_offset = max(0.01, 0.03 * y_span) if y_span > 0 else 0.02
        for point in all_points:
            ax.plot(
                [point["x_ghz"]],
                [point["y"]],
                linestyle="none",
                marker="o",
                markersize=6,
                markerfacecolor="none",
                markeredgecolor="tab:red",
                markeredgewidth=1.5,
                zorder=4,
            )
            ax.text(
                point["x_ghz"],
                point["y"] - y_text_offset,
                point["resonator_number"],
                ha="center",
                va="top",
                fontsize=8,
                color="tab:red",
                zorder=5,
            )

    def _plot_scans_panel_groups(self, scans) -> list[tuple[str, list]]:
        use_groups = self.plot_scans_group_var is not None and bool(self.plot_scans_group_var.get())
        if not use_groups:
            return [(Path(scan.filename).name, [scan]) for scan in scans]

        panels: list[tuple[str, list]] = []
        seen_groups: set[int] = set()
        for scan in scans:
            group = scan.plot_group
            if group is None:
                panels.append((Path(scan.filename).name, [scan]))
                continue
            if int(group) in seen_groups:
                continue
            grouped_scans = [s for s in scans if s.plot_group == group]
            seen_groups.add(int(group))
            panels.append((f"Group {int(group)}", grouped_scans))
        return panels

    def _plot_scans_reset_view(self) -> None:
        if self.plot_scans_figure is None or self.plot_scans_canvas is None:
            return
        scans = self._selected_scans()
        if not scans:
            return
        x_min, x_max = self._plot_scans_global_freq_range(scans)
        for ax in self.plot_scans_figure.axes:
            ax.set_xlim(x_min, x_max)
            if self.plot_scans_auto_y_var is not None and bool(self.plot_scans_auto_y_var.get()):
                self._plot_scans_autoscale_y_for_visible_x(ax)
            else:
                ax.relim()
                ax.autoscale(enable=True, axis="y", tight=False)
        self.plot_scans_canvas.draw_idle()
        if self.plot_scans_status_var is not None:
            self.plot_scans_status_var.set("Reset to full frequency range.")

    def _plot_scans_current_xlim(self) -> tuple[float, float] | None:
        if self.plot_scans_figure is None or not self.plot_scans_figure.axes:
            return None
        for ax in self.plot_scans_figure.axes:
            try:
                xlim = ax.get_xlim()
            except Exception:
                continue
            if len(xlim) == 2 and np.all(np.isfinite(np.asarray(xlim, dtype=float))):
                return float(xlim[0]), float(xlim[1])
        return None

    def _plot_scans_render(self) -> None:
        if self.plot_scans_figure is None or self.plot_scans_canvas is None:
            return

        scans = self._selected_scans()
        self._plot_scans_update_data_mode_controls()
        preferred_xlim = self._plot_scans_current_xlim()
        saved_limits = [(ax.get_xlim(), ax.get_ylim()) for ax in self.plot_scans_figure.axes]
        self.plot_scans_figure.clear()

        if not scans:
            ax = self.plot_scans_figure.add_subplot(111)
            ax.text(0.5, 0.5, "No selected scans", ha="center", va="center")
            ax.axis("off")
            self.plot_scans_canvas.draw_idle()
            return

        columns = self._plot_scans_visible_columns()
        if not columns:
            ax = self.plot_scans_figure.add_subplot(111)
            ax.text(0.5, 0.5, "No plot types enabled", ha="center", va="center")
            ax.axis("off")
            self.plot_scans_canvas.draw_idle()
            return

        panels = self._plot_scans_panel_groups(scans)
        n = len(panels)
        x_min, x_max = self._plot_scans_global_freq_range(scans)
        axes = self.plot_scans_figure.subplots(n, len(columns), sharex=True, squeeze=False)
        missing_normalized = []

        for row, (panel_title, panel_scans) in enumerate(panels):
            panel_payloads = []
            for scan in panel_scans:
                payload = self._plot_scans_series_for_scan(scan)
                if payload is None:
                    if self._plot_scans_data_mode() == "normalized":
                        missing_normalized.append(Path(scan.filename).name)
                    continue
                panel_payloads.append((scan, payload))
            col = 0
            if "amp" in columns:
                ax_amp = axes[row][col]
                used_overlay_labels: set[str] = set()
                if not panel_payloads:
                    ax_amp.text(0.5, 0.5, "No normalized data", ha="center", va="center")
                    ax_amp.set_axis_off()
                for scan, payload in panel_payloads:
                    ax_amp.plot(
                        payload["freq_ghz"],
                        payload["amp"],
                        linewidth=1.0,
                        label=Path(scan.filename).name if len(panel_payloads) > 1 else None,
                    )
                    self._plot_scans_add_overlays(
                        ax_amp,
                        scan,
                        payload,
                        use_phase=False,
                        multi_scan=len(panel_payloads) > 1,
                        used_overlay_labels=used_overlay_labels,
                    )
                if panel_payloads:
                    self._plot_scans_add_attached_resonator_overlays(
                        ax_amp,
                        panel_payloads,
                        use_phase=False,
                    )
                if panel_payloads:
                    ax_amp.set_ylabel(str(panel_payloads[0][1]["amp_label"]))
                    ax_amp.set_xlim(x_min, x_max)
                    ax_amp.grid(True, alpha=0.3)
                    ax_amp.set_title(panel_title, fontsize=9)
                    if len(ax_amp.get_legend_handles_labels()[0]) > 0:
                        ax_amp.legend(loc="upper right", fontsize=8)
                    if row == n - 1:
                        ax_amp.set_xlabel("Frequency (GHz)")
                col += 1

            if "phase" in columns:
                ax_phase = axes[row][col]
                used_overlay_labels = set()
                if not panel_payloads:
                    ax_phase.text(0.5, 0.5, "No normalized data", ha="center", va="center")
                    ax_phase.set_axis_off()
                phase_label = "Phase (deg)"
                for scan, payload in panel_payloads:
                    phase_label = str(payload["phase_label"])
                    ax_phase.plot(
                        payload["freq_ghz"],
                        payload["phase"],
                        linewidth=1.0,
                        label=Path(scan.filename).name if len(panel_payloads) > 1 else None,
                    )
                    self._plot_scans_add_overlays(
                        ax_phase,
                        scan,
                        payload,
                        use_phase=True,
                        multi_scan=len(panel_payloads) > 1,
                        used_overlay_labels=used_overlay_labels,
                    )
                if panel_payloads:
                    self._plot_scans_add_attached_resonator_overlays(
                        ax_phase,
                        panel_payloads,
                        use_phase=True,
                    )
                if panel_payloads:
                    ax_phase.set_ylabel(phase_label)
                    ax_phase.set_xlim(x_min, x_max)
                    ax_phase.grid(True, alpha=0.3)
                    if "amp" not in columns:
                        ax_phase.set_title(panel_title, fontsize=9)
                    if len(ax_phase.get_legend_handles_labels()[0]) > 0:
                        ax_phase.legend(loc="upper right", fontsize=8)
                    if row == n - 1:
                        ax_phase.set_xlabel("Frequency (GHz)")
                col += 1

        if self._plot_scans_data_mode() == "normalized" and missing_normalized:
            missing_unique = tuple(sorted(set(missing_normalized)))
            if self._plot_scans_missing_normalized_warned != missing_unique:
                self._plot_scans_missing_normalized_warned = missing_unique
                messagebox.showwarning(
                    "Missing normalized data",
                    "Normalized data does not exist for some selected scans.\n"
                    "Plotting only the scans that do have normalized data.\n\n"
                    + "\n".join(missing_unique[:10]),
                )

        if len(saved_limits) == len(self.plot_scans_figure.axes):
            for ax, (xlim, ylim) in zip(self.plot_scans_figure.axes, saved_limits):
                ax.set_xlim(xlim)
                if self.plot_scans_auto_y_var is not None and bool(self.plot_scans_auto_y_var.get()):
                    self._plot_scans_autoscale_y_for_visible_x(ax)
                else:
                    ax.set_ylim(ylim)
        else:
            for ax in self.plot_scans_figure.axes:
                if preferred_xlim is not None:
                    ax.set_xlim(preferred_xlim)
                else:
                    ax.set_xlim(x_min, x_max)
                if self.plot_scans_auto_y_var is not None and bool(self.plot_scans_auto_y_var.get()):
                    self._plot_scans_autoscale_y_for_visible_x(ax)
                else:
                    ax.relim()
                    ax.autoscale(enable=True, axis="y", tight=False)

        for ax in self.plot_scans_figure.axes:
            ax.callbacks.connect(
                "xlim_changed",
                lambda changed_ax: self._plot_scans_autoscale_y_for_visible_x(changed_ax),
            )

        shown = []
        if "amp" in columns:
            shown.append("amplitude")
        if "phase" in columns:
            shown.append("phase")
        data_mode_label = "baseline normalized data" if self._plot_scans_data_mode() == "normalized" else "raw VNA data"
        self.plot_scans_figure.suptitle(
            f"Selected VNA Scans | {data_mode_label} | showing {', '.join(shown)} | shared frequency range",
            fontsize=11,
        )
        self.plot_scans_figure.tight_layout()
        self.plot_scans_canvas.draw_idle()
        if self.plot_scans_status_var is not None:
            shown_xlim = preferred_xlim if preferred_xlim is not None else (x_min, x_max)
            status = f"Showing {len(scans)} selected scan(s) over {shown_xlim[0]:.9g} to {shown_xlim[1]:.9g} GHz."
            if self._plot_scans_data_mode() == "normalized" and missing_normalized:
                status += f" Normalized data missing for {len(set(missing_normalized))} scan(s)."
            self.plot_scans_status_var.set(status)

    def _plot_scans_save_pdf(self) -> None:
        if self.plot_scans_figure is None:
            return
        scans = self._selected_scans()
        if not scans:
            messagebox.showwarning("No selection", "No selected scans to save.")
            return
        pdf_dir = _dataset_dir(self.dataset)
        pdf_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = pdf_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_selected_vna_panels.pdf"
        self.plot_scans_figure.savefig(pdf_path)
        x_min, x_max = self._plot_scans_global_freq_range(scans)
        columns = self._plot_scans_visible_columns()
        self.dataset.processing_history.append(
            _make_event(
                "plot_selected_vna_scans",
                {
                    "selected_count": len(scans),
                    "panel_count": len(self._plot_scans_panel_groups(scans)),
                    "pdf_path": str(pdf_path),
                    "show_amplitude": "amp" in columns,
                    "show_phase": "phase" in columns,
                    "group_selected": bool(self.plot_scans_group_var.get()) if self.plot_scans_group_var is not None else False,
                    "data_mode": self._plot_scans_data_mode(),
                    "show_gaussian_candidates": bool(self.plot_scans_show_gaussian_var.get()) if self.plot_scans_show_gaussian_var is not None else False,
                    "show_dsdf_peaks": bool(self.plot_scans_show_dsdf_var.get()) if self.plot_scans_show_dsdf_var is not None else False,
                    "show_2pi_corrections": bool(self.plot_scans_show_2pi_var.get()) if self.plot_scans_show_2pi_var is not None else False,
                    "show_vna_phase_corrections": bool(self.plot_scans_show_vna_phase_var.get()) if self.plot_scans_show_vna_phase_var is not None else False,
                    "show_other_phase_discontinuities": bool(self.plot_scans_show_other_phase_var.get()) if self.plot_scans_show_other_phase_var is not None else False,
                    "show_attached_resonators": bool(self.plot_scans_show_attached_res_var.get()) if self.plot_scans_show_attached_res_var is not None else False,
                    "auto_y": bool(self.plot_scans_auto_y_var.get()) if self.plot_scans_auto_y_var is not None else False,
                    "freq_range_ghz": [x_min, x_max],
                },
            )
        )
        self._mark_dirty()
        self._log(f"Saved selected scan plot PDF: {pdf_path}")
        self._autosave_dataset()
        if self.plot_scans_status_var is not None:
            self.plot_scans_status_var.set(f"Saved PDF: {pdf_path.name}")

    def _plot_scans_close(self) -> None:
        if self.plot_scans_window is not None and self.plot_scans_window.winfo_exists():
            self.plot_scans_window.destroy()
        self.plot_scans_window = None
        self.plot_scans_canvas = None
        self.plot_scans_toolbar = None
        self.plot_scans_figure = None
        self.plot_scans_show_amp_var = None
        self.plot_scans_show_phase_var = None
        self.plot_scans_group_var = None
        self.plot_scans_data_mode_var = None
        self.plot_scans_raw_radio = None
        self.plot_scans_normalized_radio = None
        self.plot_scans_show_gaussian_var = None
        self.plot_scans_show_dsdf_var = None
        self.plot_scans_show_2pi_var = None
        self.plot_scans_show_vna_phase_var = None
        self.plot_scans_show_other_phase_var = None
        self.plot_scans_show_attached_res_var = None
        self.plot_scans_auto_y_var = None
        self.plot_scans_status_var = None
        self._plot_scans_missing_normalized_warned = None
