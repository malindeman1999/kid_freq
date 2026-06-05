from __future__ import annotations

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from matplotlib import colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter


class ResonatorDisplacementWindowMixin:
    def open_resonator_displacement_window(self) -> None:
        if self.res_displacement_window is not None and self.res_displacement_window.winfo_exists():
            self.res_displacement_window.lift()
            self._render_resonator_displacement_window()
            return

        self.res_displacement_window = tk.Toplevel(self.root)
        self.res_displacement_window.title("Resonator Relative Displacement vs Time")
        self.res_displacement_window.geometry("1380x860")
        self.res_displacement_window.protocol("WM_DELETE_WINDOW", self._close_resonator_displacement_window)

        controls = tk.Frame(self.res_displacement_window, padx=8, pady=8)
        controls.pack(side="top", fill="x")
        control_row = tk.Frame(controls)
        control_row.pack(side="top", fill="x", anchor="w")
        self.res_displacement_status_var = tk.StringVar(
            value="Showing marked resonator displacement (f - f0) / f0 versus elapsed time."
        )
        self.res_displacement_xaxis_mode_var = tk.StringVar(value="elapsed")
        self.res_displacement_initial_date_var = tk.StringVar(value=self._dataset_res_neighbor_initial_date())

        tk.Label(control_row, text="Initial Date").pack(side="left", padx=(0, 4))
        initial_date_entry = tk.Entry(control_row, width=12, textvariable=self.res_displacement_initial_date_var)
        initial_date_entry.pack(side="left", padx=(0, 4))
        initial_date_entry.bind(
            "<Return>",
            lambda _event: (self._sync_res_neighbor_initial_date(autosave=True), self._render_resonator_displacement_window()),
        )
        initial_date_entry.bind(
            "<FocusOut>",
            lambda _event: (self._sync_res_neighbor_initial_date(autosave=True), self._render_resonator_displacement_window()),
        )
        tk.Label(control_row, text="YYYY-MM-DD").pack(side="left", padx=(0, 12))
        tk.Label(control_row, text="X-Axis").pack(side="left", padx=(8, 4))
        tk.Radiobutton(
            control_row,
            text="Elapsed Time",
            value="elapsed",
            variable=self.res_displacement_xaxis_mode_var,
            command=self._render_resonator_displacement_window,
        ).pack(side="left", padx=(0, 6))
        tk.Radiobutton(
            control_row,
            text="Date",
            value="date",
            variable=self.res_displacement_xaxis_mode_var,
            command=self._render_resonator_displacement_window,
        ).pack(side="left", padx=(0, 8))
        tk.Radiobutton(
            control_row,
            text="Temperature",
            value="temperature",
            variable=self.res_displacement_xaxis_mode_var,
            command=self._render_resonator_displacement_window,
        ).pack(side="left", padx=(0, 8))
        tk.Radiobutton(
            control_row,
            text="Bias Power",
            value="power",
            variable=self.res_displacement_xaxis_mode_var,
            command=self._render_resonator_displacement_window,
        ).pack(side="left", padx=(0, 8))
        tk.Button(control_row, text="Refresh", width=10, command=self._render_resonator_displacement_window).pack(
            side="left",
            padx=(0, 8),
        )
        tk.Label(controls, textvariable=self.res_displacement_status_var, anchor="w", justify="left").pack(
            side="top",
            fill="x",
            expand=True,
            pady=(6, 0),
        )

        self.res_displacement_figure = Figure(figsize=(12.5, 7))
        self.res_displacement_canvas = FigureCanvasTkAgg(self.res_displacement_figure, master=self.res_displacement_window)
        self.res_displacement_toolbar = NavigationToolbar2Tk(self.res_displacement_canvas, self.res_displacement_window)
        self.res_displacement_toolbar.update()
        self.res_displacement_toolbar.pack(side="top", fill="x")
        self.res_displacement_canvas.get_tk_widget().pack(fill="both", expand=True)

        self._render_resonator_displacement_window()

    def _close_resonator_displacement_window(self) -> None:
        if self.res_displacement_window is not None and self.res_displacement_window.winfo_exists():
            self.res_displacement_window.destroy()
        self.res_displacement_window = None
        self.res_displacement_canvas = None
        self.res_displacement_toolbar = None
        self.res_displacement_figure = None
        self.res_displacement_status_var = None
        self.res_displacement_xaxis_mode_var = None
        self.res_displacement_initial_date_var = None
        self._res_displacement_ax = None

    def _resonator_displacement_data(self, initial_date_text: str = "") -> dict:
        tests = self._resonator_shift_test_units()
        if len(tests) < 2:
            raise ValueError("At least two selected test dates with marked resonators are required.")
        dated_tests = [test for test in tests if test.get("timestamp_dt") is not None]
        if len(dated_tests) < 2:
            raise ValueError("At least two selected test dates need valid file timestamps.")

        initial_time = self._resonator_neighbor_parse_initial_date(initial_date_text)
        base_time = initial_time if initial_time is not None else min(test["timestamp_dt"] for test in dated_tests)
        for test in tests:
            timestamp_dt = test.get("timestamp_dt")
            test["elapsed_days"] = (
                float((timestamp_dt - base_time).total_seconds()) / 86400.0
                if isinstance(timestamp_dt, datetime)
                else np.nan
            )

        values_by_resonator: dict[str, list[float]] = {}
        for test in tests:
            for resonator_label, freq_hz in test["resonators"].items():
                values_by_resonator.setdefault(str(resonator_label), []).append(float(freq_hz))
        mean_freq_by_resonator: dict[str, float] = {}
        for resonator_label, values in values_by_resonator.items():
            arr = np.asarray(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                mean_freq_by_resonator[resonator_label] = float(np.mean(arr))
        ordered_labels = sorted(
            mean_freq_by_resonator,
            key=lambda label: (mean_freq_by_resonator[label], self._resonator_sort_key(label)),
        )
        if not ordered_labels:
            raise ValueError("No marked resonators were available with finite frequencies.")

        resonator_series: list[dict] = []
        for resonator_label in ordered_labels:
            points: list[dict] = []
            for test in tests:
                elapsed_days = float(test.get("elapsed_days", np.nan))
                if not np.isfinite(elapsed_days):
                    continue
                freq_hz = test["resonators"].get(resonator_label)
                if freq_hz is None:
                    continue
                freq_hz = float(freq_hz)
                if not np.isfinite(freq_hz):
                    continue
                points.append(
                    {
                        "elapsed_days": elapsed_days,
                        "temperature_mK": test.get("temperature_mK"),
                        "bias_power_dBm": test.get("bias_power_dBm"),
                        "freq_hz": freq_hz,
                        "test_label": str(test.get("label", "")),
                    }
                )
            if len(points) < 2:
                continue
            points.sort(key=lambda item: float(item["elapsed_days"]))
            f0 = float(points[0]["freq_hz"])
            if not np.isfinite(f0) or f0 == 0.0:
                continue
            for point in points:
                point["df_over_f0"] = float((float(point["freq_hz"]) - f0) / f0)
            resonator_series.append(
                {
                    "label": str(resonator_label),
                    "mean_freq_hz": float(mean_freq_by_resonator[resonator_label]),
                    "initial_freq_hz": f0,
                    "points": points,
                }
            )

        if not resonator_series:
            raise ValueError("No resonator had at least two dated points and a finite initial frequency.")

        mean_res_freqs_hz = np.asarray([float(item["mean_freq_hz"]) for item in resonator_series], dtype=float)
        vmin = float(np.min(mean_res_freqs_hz))
        vmax = float(np.max(mean_res_freqs_hz))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            raise ValueError("Could not determine resonator mean frequencies for coloring.")
        if vmax <= vmin:
            vmax = vmin + 1.0
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.get_cmap("rainbow_r")
        for series in resonator_series:
            series["color"] = cmap(norm(float(series["mean_freq_hz"])))

        return {
            "tests": tests,
            "resonator_series": resonator_series,
            "norm": norm,
            "cmap": cmap,
            "elapsed_time_origin": base_time,
        }

    def _render_resonator_displacement_window(self) -> None:
        if self.res_displacement_figure is None or self.res_displacement_canvas is None:
            return
        self.res_displacement_figure.clear()
        ax = self.res_displacement_figure.add_subplot(111)
        self._res_displacement_ax = ax

        initial_date_text = (
            str(self.res_displacement_initial_date_var.get())
            if self.res_displacement_initial_date_var is not None
            else ""
        )
        try:
            state = self._resonator_displacement_data(initial_date_text=initial_date_text)
        except Exception as exc:
            ax.text(0.5, 0.5, str(exc), ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            if self.res_displacement_status_var is not None:
                self.res_displacement_status_var.set(str(exc))
            self.res_displacement_canvas.draw_idle()
            return

        resonator_series = state["resonator_series"]
        xaxis_mode = (
            str(self.res_displacement_xaxis_mode_var.get())
            if self.res_displacement_xaxis_mode_var is not None
            else "elapsed"
        ).strip().lower()
        if xaxis_mode not in {"elapsed", "date", "temperature", "power"}:
            xaxis_mode = "elapsed"
        x_key = (
            "temperature_mK"
            if xaxis_mode == "temperature"
            else "bias_power_dBm"
            if xaxis_mode == "power"
            else "elapsed_days"
        )

        if xaxis_mode == "temperature":
            missing_temperature = [
                str(test.get("label", test.get("detail_label", "unknown scan")))
                for test in state["tests"]
                if test.get("temperature_mK") is None or bool(test.get("missing_temperature", False))
            ]
            if missing_temperature:
                message = (
                    "Temperature x-axis requires every selected VNA scan in the plotted test units "
                    "to have temperature_mK assigned. Missing temperature for: "
                    + ", ".join(missing_temperature[:8])
                    + (f", ... (+{len(missing_temperature) - 8} more)" if len(missing_temperature) > 8 else "")
                )
                ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, wrap=True)
                ax.set_axis_off()
                if self.res_displacement_status_var is not None:
                    self.res_displacement_status_var.set(message)
                self.res_displacement_canvas.draw_idle()
                return
        if xaxis_mode == "power":
            missing_power = [
                str(test.get("label", test.get("detail_label", "unknown scan")))
                for test in state["tests"]
                if test.get("bias_power_dBm") is None or bool(test.get("missing_bias_power", False))
            ]
            if missing_power:
                message = (
                    "Bias-power x-axis requires every selected VNA scan in the plotted test units "
                    "to have bias_power_dBm assigned. Missing bias power for: "
                    + ", ".join(missing_power[:8])
                    + (f", ... (+{len(missing_power) - 8} more)" if len(missing_power) > 8 else "")
                )
                ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, wrap=True)
                ax.set_axis_off()
                if self.res_displacement_status_var is not None:
                    self.res_displacement_status_var.set(message)
                self.res_displacement_canvas.draw_idle()
                return

        def _plot_x(point: dict) -> float:
            try:
                return float(point.get(x_key, np.nan))
            except Exception:
                return np.nan

        values_by_x: dict[float, list[float]] = {}
        for series in resonator_series:
            points = sorted(series["points"], key=_plot_x)
            x = np.asarray([_plot_x(pt) for pt in points], dtype=float)
            y = np.asarray([float(pt["df_over_f0"]) for pt in points], dtype=float)
            ax.plot(x, y, color=series["color"], linewidth=1.4, alpha=0.9, marker="o", markersize=3.5, zorder=3.0)
            for x_val, y_val in zip(x, y):
                if np.isfinite(x_val) and np.isfinite(y_val):
                    values_by_x.setdefault(float(x_val), []).append(float(y_val))

        summary_x: list[float] = []
        summary_mean: list[float] = []
        summary_low: list[float] = []
        summary_high: list[float] = []
        final_displacements: list[float] = []
        for x_value in sorted(values_by_x):
            arr = np.asarray(values_by_x[x_value], dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            mean_value = float(np.mean(arr))
            std_value = float(np.std(arr))
            summary_x.append(float(x_value))
            summary_mean.append(mean_value)
            summary_low.append(mean_value - std_value)
            summary_high.append(mean_value + std_value)
        for series in resonator_series:
            points = sorted(series.get("points", []), key=_plot_x)
            if not points:
                continue
            y_final = float(points[-1].get("df_over_f0", np.nan))
            if np.isfinite(y_final):
                final_displacements.append(y_final)
        mean_net_shift = (
            float(np.mean(np.asarray(final_displacements, dtype=float)))
            if final_displacements
            else np.nan
        )

        if summary_x:
            x_arr = np.asarray(summary_x, dtype=float)
            low_arr = np.asarray(summary_low, dtype=float)
            high_arr = np.asarray(summary_high, dtype=float)
            mean_arr = np.asarray(summary_mean, dtype=float)
            ax.fill_between(
                x_arr,
                low_arr,
                high_arr,
                color="0.75",
                alpha=0.65,
                zorder=1.2,
                linewidth=0.0,
                label="Mean +/- 1 std",
            )
            ax.plot(x_arr, mean_arr, color="black", linewidth=2.4, alpha=1.0, zorder=4.2, label="Mean")
            ax.legend(loc="best", fontsize=8)

        ax.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--")
        ax.grid(True, alpha=0.3)
        origin_dt = state.get("elapsed_time_origin")
        if xaxis_mode == "date" and isinstance(origin_dt, datetime):
            ax.set_xlabel("Date")
            ax.xaxis.set_major_formatter(
                FuncFormatter(
                    lambda value, _pos: (origin_dt + timedelta(days=float(value))).strftime("%Y-%m-%d")
                    if np.isfinite(value)
                    else ""
                )
            )
            ax.tick_params(axis="x", labelrotation=30)
        elif xaxis_mode == "temperature":
            ax.set_xlabel("Temperature (mK)")
        elif xaxis_mode == "power":
            ax.set_xlabel("Bias Power (dBm)")
        else:
            ax.set_xlabel("Elapsed Time (days)")
        ax.set_ylabel("Relative Displacement (f - f0) / f0")
        ax.set_title(
            "Marked Resonator Relative Displacement vs Temperature"
            if xaxis_mode == "temperature"
            else "Marked Resonator Relative Displacement vs Bias Power"
            if xaxis_mode == "power"
            else "Marked Resonator Relative Displacement vs Time"
        )
        mean_shift_label = (
            f"Mean net shift (final points): {mean_net_shift:+.3e} df/f"
            if np.isfinite(mean_net_shift)
            else "Mean net shift (final points): n/a"
        )
        ax.text(
            0.02,
            0.98,
            mean_shift_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.6"},
        )

        sm = plt.cm.ScalarMappable(norm=state["norm"], cmap=state["cmap"])
        sm.set_array([])
        colorbar = self.res_displacement_figure.colorbar(sm, ax=ax, pad=0.02)
        colorbar.set_label("Resonator Mean Frequency (GHz)")
        colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value / 1.0e9:.3f}"))

        self.res_displacement_figure.tight_layout()
        if self.res_displacement_status_var is not None:
            origin_text = origin_dt.strftime("%Y-%m-%d") if isinstance(origin_dt, datetime) else "unknown"
            if xaxis_mode == "temperature":
                origin_prefix = "X-axis: VNA temperature_mK. "
            elif xaxis_mode == "power":
                origin_prefix = "X-axis: VNA bias_power_dBm. "
            else:
                origin_prefix = f"Elapsed-time origin: {origin_text}. "
            self.res_displacement_status_var.set(
                f"{origin_prefix}Showing {len(resonator_series)} resonator curve(s) from {len(state['tests'])} selected test unit(s); mean net shift (final points) = {mean_net_shift:+.3e} df/f. Grey band is mean +/- 1 std and black curve is mean."
            )
        self.res_displacement_canvas.draw_idle()
