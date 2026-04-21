from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy import stats

class ResonatorShiftHistMixin:
    def _resonator_shift_correlation_data(self) -> dict:
        tests = self._resonator_shift_test_units()
        if len(tests) < 3:
            raise ValueError("At least three selected test dates with marked resonators are required for correlation analysis.")

        values_by_resonator: dict[str, list[float]] = {}
        for test in tests:
            for resonator_label, freq_hz in test["resonators"].items():
                values_by_resonator.setdefault(resonator_label, []).append(float(freq_hz))

        mean_freq_by_resonator: dict[str, float] = {}
        for resonator_label, values in values_by_resonator.items():
            arr = np.asarray(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                mean_freq_by_resonator[resonator_label] = float(np.mean(arr))

        pair_count = len(tests) - 1
        resonator_labels = sorted(mean_freq_by_resonator, key=lambda label: (mean_freq_by_resonator[label], self._resonator_sort_key(label)))
        if not resonator_labels:
            raise ValueError("No marked resonators were found for the selected tests.")

        label_to_index = {label: idx for idx, label in enumerate(resonator_labels)}
        shift_matrix = np.full((len(resonator_labels), pair_count), np.nan, dtype=float)
        pair_labels: list[str] = []
        for pair_idx in range(pair_count):
            left = tests[pair_idx]
            right = tests[pair_idx + 1]
            pair_labels.append(f"{left['date_label']} -> {right['date_label']}")
            shared = set(left["resonators"]).intersection(right["resonators"])
            for resonator_label in shared:
                row_idx = label_to_index.get(resonator_label)
                mean_freq_hz = mean_freq_by_resonator.get(resonator_label)
                if row_idx is None or mean_freq_hz is None or mean_freq_hz == 0.0:
                    continue
                delta_rel = (float(right["resonators"][resonator_label]) - float(left["resonators"][resonator_label])) / float(mean_freq_hz)
                shift_matrix[row_idx, pair_idx] = float(delta_rel)

        valid_counts = np.sum(np.isfinite(shift_matrix), axis=1)
        keep_mask = valid_counts >= 2
        if np.count_nonzero(keep_mask) < 2:
            raise ValueError("Need at least two resonators with shifts on two or more consecutive test pairs.")
        resonator_labels = [label for label, keep in zip(resonator_labels, keep_mask) if keep]
        mean_freqs_hz = np.asarray([mean_freq_by_resonator[label] for label in resonator_labels], dtype=float)
        shift_matrix = shift_matrix[keep_mask, :]

        corr_matrix = np.full((len(resonator_labels), len(resonator_labels)), np.nan, dtype=float)
        pair_points: list[dict] = []
        for i in range(len(resonator_labels)):
            corr_matrix[i, i] = 1.0
            for j in range(i + 1, len(resonator_labels)):
                xi = shift_matrix[i, :]
                xj = shift_matrix[j, :]
                mask = np.isfinite(xi) & np.isfinite(xj)
                if np.count_nonzero(mask) < 2:
                    continue
                vi = xi[mask]
                vj = xj[mask]
                std_i = float(np.std(vi))
                std_j = float(np.std(vj))
                if std_i == 0.0 and std_j == 0.0:
                    corr = 1.0 if np.allclose(vi, vj) else np.nan
                elif std_i == 0.0 or std_j == 0.0:
                    corr = np.nan
                else:
                    corr = float(np.corrcoef(vi, vj)[0, 1])
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
                if np.isfinite(corr):
                    pair_points.append(
                        {
                            "separation_hz": float(abs(mean_freqs_hz[j] - mean_freqs_hz[i])),
                            "correlation": corr,
                        }
                    )

        if not pair_points:
            raise ValueError("No resonator pairs had enough overlapping shift measurements to correlate.")

        separations_hz = np.asarray([pt["separation_hz"] for pt in pair_points], dtype=float)
        max_sep = float(np.max(separations_hz))
        if max_sep <= 0.0:
            bin_edges = np.asarray([0.0, 1.0], dtype=float)
        else:
            nbins = min(16, max(4, int(np.sqrt(len(pair_points)))))
            bin_edges = np.linspace(0.0, max_sep, nbins + 1)
            if not np.all(np.diff(bin_edges) > 0):
                bin_edges = np.asarray([0.0, max_sep], dtype=float)
        bin_centers: list[float] = []
        bin_means: list[float] = []
        bin_counts: list[int] = []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            if hi <= lo:
                continue
            if hi == bin_edges[-1]:
                mask = (separations_hz >= lo) & (separations_hz <= hi)
            else:
                mask = (separations_hz >= lo) & (separations_hz < hi)
            if not np.any(mask):
                continue
            corr_vals = np.asarray([pair_points[idx]["correlation"] for idx in np.flatnonzero(mask)], dtype=float)
            corr_vals = corr_vals[np.isfinite(corr_vals)]
            if corr_vals.size == 0:
                continue
            bin_centers.append(0.5 * (lo + hi))
            bin_means.append(float(np.mean(corr_vals)))
            bin_counts.append(int(corr_vals.size))

        return {
            "tests": tests,
            "pair_labels": pair_labels,
            "resonator_labels": resonator_labels,
            "mean_freqs_hz": mean_freqs_hz,
            "shift_matrix": shift_matrix,
            "corr_matrix": corr_matrix,
            "pair_points": pair_points,
            "bin_centers_hz": np.asarray(bin_centers, dtype=float),
            "bin_means": np.asarray(bin_means, dtype=float),
            "bin_counts": np.asarray(bin_counts, dtype=int),
        }



    def open_resonator_shift_correlation_window(self) -> None:
        if self.res_shift_corr_window is not None and self.res_shift_corr_window.winfo_exists():
            self.res_shift_corr_window.lift()
            self._render_resonator_shift_correlation_window()
            return

        self.res_shift_corr_window = tk.Toplevel(self.root)
        self.res_shift_corr_window.title("Resonator Shift Correlation")
        self.res_shift_corr_window.geometry("1460x900")
        self.res_shift_corr_window.protocol("WM_DELETE_WINDOW", self._close_resonator_shift_correlation_window)

        controls = tk.Frame(self.res_shift_corr_window, padx=8, pady=8)
        controls.pack(side="top", fill="x")
        self.res_shift_corr_status_var = tk.StringVar(
            value="Showing correlation of resonator df/f across consecutive selected tests."
        )
        tk.Label(controls, textvariable=self.res_shift_corr_status_var, anchor="w").pack(
            side="left", fill="x", expand=True
        )
        tk.Button(controls, text="Refresh", width=10, command=self._render_resonator_shift_correlation_window).pack(
            side="right"
        )

        self.res_shift_corr_figure = Figure(figsize=(13, 8))
        self.res_shift_corr_canvas = FigureCanvasTkAgg(self.res_shift_corr_figure, master=self.res_shift_corr_window)
        self.res_shift_corr_toolbar = NavigationToolbar2Tk(self.res_shift_corr_canvas, self.res_shift_corr_window)
        self.res_shift_corr_toolbar.update()
        self.res_shift_corr_toolbar.pack(side="top", fill="x")
        self.res_shift_corr_canvas.get_tk_widget().pack(fill="both", expand=True)

        self._render_resonator_shift_correlation_window()



    def _close_resonator_shift_correlation_window(self) -> None:
        if self.res_shift_corr_window is not None and self.res_shift_corr_window.winfo_exists():
            self.res_shift_corr_window.destroy()
        self.res_shift_corr_window = None
        self.res_shift_corr_canvas = None
        self.res_shift_corr_toolbar = None
        self.res_shift_corr_figure = None
        self.res_shift_corr_status_var = None
        self._res_shift_corr_axes = None



    def _render_resonator_shift_correlation_window(self) -> None:
        if self.res_shift_corr_figure is None or self.res_shift_corr_canvas is None:
            return

        prior_scatter_xlim = None
        prior_scatter_ylim = None
        if self._res_shift_corr_axes is not None:
            try:
                prior_scatter_xlim = self._res_shift_corr_axes[1].get_xlim()
                prior_scatter_ylim = self._res_shift_corr_axes[1].get_ylim()
            except Exception:
                prior_scatter_xlim = None
                prior_scatter_ylim = None

        self.res_shift_corr_figure.clear()
        ax_heat = self.res_shift_corr_figure.add_subplot(1, 2, 1)
        ax_scatter = self.res_shift_corr_figure.add_subplot(1, 2, 2)
        self._res_shift_corr_axes = (ax_heat, ax_scatter)

        try:
            data = self._resonator_shift_correlation_data()
        except Exception as exc:
            ax_heat.text(0.5, 0.5, str(exc), ha="center", va="center", transform=ax_heat.transAxes)
            ax_heat.set_axis_off()
            ax_scatter.set_axis_off()
            if self.res_shift_corr_status_var is not None:
                self.res_shift_corr_status_var.set(str(exc))
            self.res_shift_corr_canvas.draw_idle()
            return

        mean_freqs_ghz = np.asarray(data["mean_freqs_hz"], dtype=float) / 1.0e9
        corr_matrix = np.asarray(data["corr_matrix"], dtype=float)
        pair_points = data["pair_points"]
        bin_centers_mhz = np.asarray(data["bin_centers_hz"], dtype=float) / 1.0e6
        bin_means = np.asarray(data["bin_means"], dtype=float)

        freq_min = float(np.min(mean_freqs_ghz))
        freq_max = float(np.max(mean_freqs_ghz))
        im = ax_heat.imshow(
            corr_matrix,
            origin="lower",
            aspect="auto",
            extent=[freq_min, freq_max, freq_min, freq_max],
            vmin=-1.0,
            vmax=1.0,
            cmap="coolwarm",
        )
        ax_heat.set_xlabel("Mean Resonator Frequency (GHz)")
        ax_heat.set_ylabel("Mean Resonator Frequency (GHz)")
        ax_heat.set_title("Shift Correlation Heatmap")
        colorbar = self.res_shift_corr_figure.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
        colorbar.set_label("Correlation")

        sep_mhz = np.asarray([float(pt["separation_hz"]) / 1.0e6 for pt in pair_points], dtype=float)
        corr_vals = np.asarray([float(pt["correlation"]) for pt in pair_points], dtype=float)
        ax_scatter.scatter(
            sep_mhz,
            corr_vals,
            s=18,
            color="tab:blue",
            alpha=0.45,
            edgecolors="none",
            label="Resonator pairs",
        )
        if bin_centers_mhz.size and bin_means.size:
            ax_scatter.plot(
                bin_centers_mhz,
                bin_means,
                color="black",
                linewidth=2.0,
                marker="o",
                markersize=4,
                label="Binned mean",
            )
        ax_scatter.axhline(0.0, color="0.4", linewidth=0.8, linestyle="--")
        ax_scatter.grid(True, alpha=0.3)
        ax_scatter.set_xlabel("Resonator Frequency Separation (MHz)")
        ax_scatter.set_ylabel("Shift Correlation")
        ax_scatter.set_title("Correlation vs Frequency Separation")
        ax_scatter.legend(loc="best", fontsize=8)

        if prior_scatter_xlim is not None and prior_scatter_ylim is not None:
            ax_scatter.set_xlim(prior_scatter_xlim)
            ax_scatter.set_ylim(prior_scatter_ylim)

        self.res_shift_corr_figure.suptitle("Resonator Shift Correlations Across Tests", fontsize=12)
        self.res_shift_corr_figure.tight_layout()
        if self.res_shift_corr_status_var is not None:
            self.res_shift_corr_status_var.set(
                f"Showing {len(data['resonator_labels'])} resonators and {len(pair_points)} resonator-pair correlations across {len(data['pair_labels'])} consecutive test pair(s)."
            )
        self.res_shift_corr_canvas.draw_idle()



    def _resonator_pair_dfdiff_hist_data(self, max_separation_mhz: float) -> dict:
        tests = self._resonator_shift_test_units()
        if len(tests) < 2:
            raise ValueError("At least two selected test dates with marked resonators are required.")

        max_separation_hz = max(0.0, float(max_separation_mhz)) * 1.0e6

        values_by_resonator: dict[str, list[float]] = {}
        for test in tests:
            for resonator_label, freq_hz in test["resonators"].items():
                values_by_resonator.setdefault(resonator_label, []).append(float(freq_hz))

        mean_freq_by_resonator: dict[str, float] = {}
        for resonator_label, values in values_by_resonator.items():
            arr = np.asarray(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                mean_freq_by_resonator[resonator_label] = float(np.mean(arr))
        if not mean_freq_by_resonator:
            raise ValueError("No marked resonators were found for the selected tests.")

        pair_labels: list[str] = []
        delta_diff_records: list[dict] = []
        for pair_idx in range(len(tests) - 1):
            left = tests[pair_idx]
            right = tests[pair_idx + 1]
            pair_label = f"{left['date_label']} -> {right['date_label']}"
            pair_labels.append(pair_label)

            shared_labels = [
                label
                for label in left["resonators"]
                if label in right["resonators"] and label in mean_freq_by_resonator
            ]
            shared_labels.sort(
                key=lambda label: (float(mean_freq_by_resonator[label]), self._resonator_sort_key(label))
            )
            if len(shared_labels) < 2:
                continue

            deltas_hz: dict[str, float] = {}
            for label in shared_labels:
                delta_hz = float(right["resonators"][label]) - float(left["resonators"][label])
                if np.isfinite(delta_hz):
                    deltas_hz[label] = delta_hz

            ordered = [label for label in shared_labels if label in deltas_hz]
            for left_idx in range(len(ordered) - 1):
                label1 = ordered[left_idx]
                f1_hz = float(mean_freq_by_resonator[label1])
                df1_hz = float(deltas_hz[label1])
                for right_idx in range(left_idx + 1, len(ordered)):
                    label2 = ordered[right_idx]
                    f2_hz = float(mean_freq_by_resonator[label2])
                    separation_hz = f2_hz - f1_hz
                    if separation_hz < 0.0:
                        continue
                    if separation_hz > max_separation_hz:
                        break
                    left_f1_hz = float(left["resonators"][label1])
                    left_f2_hz = float(left["resonators"][label2])
                    right_f1_hz = float(right["resonators"][label1])
                    right_f2_hz = float(right["resonators"][label2])
                    df2_hz = float(deltas_hz[label2])
                    rel_delta_diff = (df2_hz - df1_hz) / f1_hz if f1_hz != 0.0 else np.nan
                    start_rel_sep = (left_f2_hz - left_f1_hz) / left_f1_hz if left_f1_hz != 0.0 else np.nan
                    end_rel_sep = (right_f2_hz - right_f1_hz) / right_f1_hz if right_f1_hz != 0.0 else np.nan
                    delta_diff_records.append(
                        {
                            "pair_idx": pair_idx,
                            "pair_label": pair_label,
                            "res1": label1,
                            "res2": label2,
                            "f1_hz": f1_hz,
                            "f2_hz": f2_hz,
                            "left_f1_hz": left_f1_hz,
                            "left_f2_hz": left_f2_hz,
                            "right_f1_hz": right_f1_hz,
                            "right_f2_hz": right_f2_hz,
                            "separation_hz": separation_hz,
                            "df1_hz": df1_hz,
                            "df2_hz": df2_hz,
                            "delta_diff_hz": df2_hz - df1_hz,
                            "rel_delta_diff": rel_delta_diff,
                            "start_rel_sep": start_rel_sep,
                            "end_rel_sep": end_rel_sep,
                        }
                    )

        if not delta_diff_records:
            raise ValueError(
                f"No resonator pairs within {max_separation_mhz:.3g} MHz were found across consecutive selected tests."
            )

        return {
            "tests": tests,
            "pair_labels": pair_labels,
            "records": delta_diff_records,
        }



    def open_resonator_pair_dfdiff_hist_window(self) -> None:
        if self.res_pair_dfdiff_hist_window is not None and self.res_pair_dfdiff_hist_window.winfo_exists():
            self.res_pair_dfdiff_hist_window.lift()
            self._render_resonator_pair_dfdiff_hist_window()
            return

        self.res_pair_dfdiff_hist_window = tk.Toplevel(self.root)
        self.res_pair_dfdiff_hist_window.title("Histogram of df2-df1")
        self.res_pair_dfdiff_hist_window.geometry("1320x860")
        self.res_pair_dfdiff_hist_window.protocol("WM_DELETE_WINDOW", self._close_resonator_pair_dfdiff_hist_window)

        controls = tk.Frame(self.res_pair_dfdiff_hist_window, padx=8, pady=8)
        controls.pack(side="top", fill="x")
        self.res_pair_dfdiff_hist_status_var = tk.StringVar(
            value="Showing counts of (df2-df1)/f1 for resonator pairs from consecutive selected tests."
        )
        self.res_pair_dfdiff_hist_sep_mhz_var = tk.DoubleVar(value=10.0)
        self.res_pair_dfdiff_hist_bin_mhz_var = tk.DoubleVar(value=0.0001)
        self.res_pair_dfdiff_hist_capture_var = tk.DoubleVar(value=1.0 / 20000.0)
        self.res_pair_dfdiff_hist_fit_mode_var = tk.StringVar(value="gennorm")
        self.res_pair_dfdiff_hist_num_res_var = tk.IntVar(value=1000)
        self.res_pair_dfdiff_hist_center_ghz_var = tk.DoubleVar(value=0.8)
        self.res_pair_dfdiff_hist_dfrel_var = tk.DoubleVar(value=1.8e-3)
        self.res_pair_dfdiff_hist_freq_jitter_khz_var = tk.DoubleVar(value=0.0)

        scale_wrap = tk.Frame(controls)
        scale_wrap.pack(side="top", fill="x")
        tk.Label(scale_wrap, text="N").pack(side="left", padx=(0, 4))
        num_res_entry = tk.Entry(scale_wrap, width=7, textvariable=self.res_pair_dfdiff_hist_num_res_var)
        num_res_entry.pack(side="left")
        num_res_entry.bind("<Return>", lambda _event: self._render_resonator_pair_dfdiff_hist_window())
        num_res_entry.bind("<FocusOut>", lambda _event: self._render_resonator_pair_dfdiff_hist_window())
        tk.Label(scale_wrap, text="Center (GHz)").pack(side="left", padx=(10, 4))
        center_entry = tk.Entry(scale_wrap, width=6, textvariable=self.res_pair_dfdiff_hist_center_ghz_var)
        center_entry.pack(side="left")
        center_entry.bind("<Return>", lambda _event: self._render_resonator_pair_dfdiff_hist_window())
        center_entry.bind("<FocusOut>", lambda _event: self._render_resonator_pair_dfdiff_hist_window())
        tk.Label(scale_wrap, text="df/f").pack(side="left", padx=(10, 4))
        dfrel_entry = tk.Entry(scale_wrap, width=8, textvariable=self.res_pair_dfdiff_hist_dfrel_var)
        dfrel_entry.pack(side="left")
        dfrel_entry.bind("<Return>", lambda _event: self._render_resonator_pair_dfdiff_hist_window())
        dfrel_entry.bind("<FocusOut>", lambda _event: self._render_resonator_pair_dfdiff_hist_window())
        tk.Button(
            scale_wrap, text="Refresh", width=10, command=self._render_resonator_pair_dfdiff_hist_window
        ).pack(side="right", padx=(8, 0))
        tk.Label(controls, textvariable=self.res_pair_dfdiff_hist_status_var, anchor="w", justify="left").pack(
            side="top", fill="x", pady=(6, 0)
        )
        tk.Label(scale_wrap, text="Max |f2-f1| (MHz)").pack(side="left", padx=(0, 4))
        self.res_pair_dfdiff_hist_sep_scale = tk.Scale(
            scale_wrap,
            from_=0.1,
            to=100.0,
            resolution=0.1,
            orient="horizontal",
            length=170,
            showvalue=True,
            variable=self.res_pair_dfdiff_hist_sep_mhz_var,
        )
        self.res_pair_dfdiff_hist_sep_scale.pack(side="left")
        self.res_pair_dfdiff_hist_sep_scale.bind(
            "<ButtonRelease-1>",
            lambda _event: self._render_resonator_pair_dfdiff_hist_window(),
        )
        self.res_pair_dfdiff_hist_sep_scale.bind(
            "<KeyRelease>",
            lambda _event: self._render_resonator_pair_dfdiff_hist_window(),
        )
        tk.Label(scale_wrap, text="Bin width").pack(side="left", padx=(10, 4))
        self.res_pair_dfdiff_hist_bin_scale = tk.Scale(
            scale_wrap,
            from_=0.00001,
            to=5.0,
            resolution=0.00001,
            orient="horizontal",
            length=160,
            showvalue=True,
            variable=self.res_pair_dfdiff_hist_bin_mhz_var,
        )
        self.res_pair_dfdiff_hist_bin_scale.pack(side="left")
        self.res_pair_dfdiff_hist_bin_scale.bind(
            "<ButtonRelease-1>",
            lambda _event: self._render_resonator_pair_dfdiff_hist_window(),
        )
        self.res_pair_dfdiff_hist_bin_scale.bind(
            "<KeyRelease>",
            lambda _event: self._render_resonator_pair_dfdiff_hist_window(),
        )
        tk.Label(scale_wrap, text="Collision +/-").pack(side="left", padx=(10, 4))
        self.res_pair_dfdiff_hist_capture_scale = tk.Scale(
            scale_wrap,
            from_=0.00001,
            to=0.0005,
            resolution=0.00001,
            orient="horizontal",
            length=160,
            showvalue=True,
            variable=self.res_pair_dfdiff_hist_capture_var,
        )
        self.res_pair_dfdiff_hist_capture_scale.pack(side="left")
        self.res_pair_dfdiff_hist_capture_scale.bind(
            "<ButtonRelease-1>",
            lambda _event: self._render_resonator_pair_dfdiff_hist_window(),
        )
        self.res_pair_dfdiff_hist_capture_scale.bind(
            "<KeyRelease>",
            lambda _event: self._render_resonator_pair_dfdiff_hist_window(),
        )
        tk.Label(scale_wrap, text="Offset std (kHz)").pack(side="left", padx=(10, 4))
        self.res_pair_dfdiff_hist_freq_jitter_scale = tk.Scale(
            scale_wrap,
            from_=0.0,
            to=1000.0,
            resolution=1.0,
            orient="horizontal",
            length=160,
            showvalue=True,
            variable=self.res_pair_dfdiff_hist_freq_jitter_khz_var,
        )
        self.res_pair_dfdiff_hist_freq_jitter_scale.pack(side="left")
        self.res_pair_dfdiff_hist_freq_jitter_scale.bind(
            "<ButtonRelease-1>",
            lambda _event: self._render_resonator_pair_dfdiff_hist_window(),
        )
        self.res_pair_dfdiff_hist_freq_jitter_scale.bind(
            "<KeyRelease>",
            lambda _event: self._render_resonator_pair_dfdiff_hist_window(),
        )
        tk.Label(scale_wrap, text="Fit").pack(side="left", padx=(10, 4))
        tk.Radiobutton(
            scale_wrap,
            text="Gaussian",
            value="gaussian",
            variable=self.res_pair_dfdiff_hist_fit_mode_var,
            command=self._render_resonator_pair_dfdiff_hist_window,
        ).pack(side="left")
        tk.Radiobutton(
            scale_wrap,
            text="Gen. normal",
            value="gennorm",
            variable=self.res_pair_dfdiff_hist_fit_mode_var,
            command=self._render_resonator_pair_dfdiff_hist_window,
        ).pack(side="left")

        self.res_pair_dfdiff_hist_figure = Figure(figsize=(12, 7))
        self.res_pair_dfdiff_hist_canvas = FigureCanvasTkAgg(
            self.res_pair_dfdiff_hist_figure,
            master=self.res_pair_dfdiff_hist_window,
        )
        self.res_pair_dfdiff_hist_toolbar = NavigationToolbar2Tk(
            self.res_pair_dfdiff_hist_canvas,
            self.res_pair_dfdiff_hist_window,
        )
        self.res_pair_dfdiff_hist_toolbar.update()
        self.res_pair_dfdiff_hist_toolbar.pack(side="top", fill="x")
        self.res_pair_dfdiff_hist_canvas.get_tk_widget().pack(fill="both", expand=True)

        self._render_resonator_pair_dfdiff_hist_window()



    def _close_resonator_pair_dfdiff_hist_window(self) -> None:
        if self.res_pair_dfdiff_hist_window is not None and self.res_pair_dfdiff_hist_window.winfo_exists():
            self.res_pair_dfdiff_hist_window.destroy()
        self.res_pair_dfdiff_hist_window = None
        self.res_pair_dfdiff_hist_canvas = None
        self.res_pair_dfdiff_hist_toolbar = None
        self.res_pair_dfdiff_hist_figure = None
        self.res_pair_dfdiff_hist_status_var = None
        self.res_pair_dfdiff_hist_sep_mhz_var = None
        self.res_pair_dfdiff_hist_bin_mhz_var = None
        self.res_pair_dfdiff_hist_capture_var = None
        self.res_pair_dfdiff_hist_fit_mode_var = None
        self.res_pair_dfdiff_hist_num_res_var = None
        self.res_pair_dfdiff_hist_center_ghz_var = None
        self.res_pair_dfdiff_hist_dfrel_var = None
        self.res_pair_dfdiff_hist_freq_jitter_khz_var = None
        self.res_pair_dfdiff_hist_sep_scale = None
        self.res_pair_dfdiff_hist_bin_scale = None
        self.res_pair_dfdiff_hist_capture_scale = None
        self.res_pair_dfdiff_hist_freq_jitter_scale = None
        self._res_pair_dfdiff_hist_ax = None



    def _render_resonator_pair_dfdiff_hist_window(self) -> None:
        if self.res_pair_dfdiff_hist_figure is None or self.res_pair_dfdiff_hist_canvas is None:
            return

        prior_xlim = None
        prior_ylim = None
        if self._res_pair_dfdiff_hist_ax is not None:
            try:
                prior_xlim = self._res_pair_dfdiff_hist_ax.get_xlim()
                prior_ylim = self._res_pair_dfdiff_hist_ax.get_ylim()
            except Exception:
                prior_xlim = None
                prior_ylim = None

        self.res_pair_dfdiff_hist_figure.clear()
        gs = self.res_pair_dfdiff_hist_figure.add_gridspec(
            2,
            2,
            width_ratios=[3.0, 2.0],
            height_ratios=[3.0, 2.0],
        )
        ax_hist = self.res_pair_dfdiff_hist_figure.add_subplot(gs[0, 0])
        ax_prob = self.res_pair_dfdiff_hist_figure.add_subplot(gs[0, 1])
        ax_cdf = self.res_pair_dfdiff_hist_figure.add_subplot(gs[1, :])
        ax = ax_hist
        self._res_pair_dfdiff_hist_ax = ax_hist

        max_sep_mhz = (
            float(self.res_pair_dfdiff_hist_sep_mhz_var.get())
            if self.res_pair_dfdiff_hist_sep_mhz_var is not None
            else 10.0
        )
        bin_width_mhz = (
            float(self.res_pair_dfdiff_hist_bin_mhz_var.get())
            if self.res_pair_dfdiff_hist_bin_mhz_var is not None
            else 0.1
        )
        bin_width_mhz = max(bin_width_mhz, 1.0e-6)
        capture_threshold = (
            float(self.res_pair_dfdiff_hist_capture_var.get())
            if self.res_pair_dfdiff_hist_capture_var is not None
            else (1.0 / 20000.0)
        )
        capture_threshold = max(capture_threshold, 0.0)
        num_resonators = (
            max(2, int(self.res_pair_dfdiff_hist_num_res_var.get()))
            if self.res_pair_dfdiff_hist_num_res_var is not None
            else 1000
        )
        center_ghz = (
            float(self.res_pair_dfdiff_hist_center_ghz_var.get())
            if self.res_pair_dfdiff_hist_center_ghz_var is not None
            else 0.8
        )
        if not np.isfinite(center_ghz) or center_ghz <= 0.0:
            center_ghz = 0.8
        dfrel_nominal = (
            float(self.res_pair_dfdiff_hist_dfrel_var.get())
            if self.res_pair_dfdiff_hist_dfrel_var is not None
            else 1.8e-3
        )
        if not np.isfinite(dfrel_nominal) or dfrel_nominal <= 0.0:
            dfrel_nominal = 1.8e-3
        freq_jitter_khz = (
            float(self.res_pair_dfdiff_hist_freq_jitter_khz_var.get())
            if self.res_pair_dfdiff_hist_freq_jitter_khz_var is not None
            else 0.0
        )
        freq_jitter_hz = max(0.0, freq_jitter_khz) * 1.0e3

        try:
            data = self._resonator_pair_dfdiff_hist_data(max_sep_mhz)
        except Exception as exc:
            ax.text(0.5, 0.5, str(exc), ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            ax_prob.set_axis_off()
            ax_cdf.set_axis_off()
            if self.res_pair_dfdiff_hist_status_var is not None:
                self.res_pair_dfdiff_hist_status_var.set(str(exc))
            self.res_pair_dfdiff_hist_canvas.draw_idle()
            return

        values_mhz = np.asarray(
            [float(record["rel_delta_diff"]) for record in data["records"]],
            dtype=float,
        )
        values_mhz = values_mhz[np.isfinite(values_mhz)]
        if values_mhz.size == 0:
            ax.text(0.5, 0.5, "No finite (df2-df1)/f1 values were available.", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            if self.res_pair_dfdiff_hist_status_var is not None:
                self.res_pair_dfdiff_hist_status_var.set("No finite (df2-df1)/f1 values were available.")
            self.res_pair_dfdiff_hist_canvas.draw_idle()
            return

        vmin = float(np.min(values_mhz))
        vmax = float(np.max(values_mhz))
        if np.isclose(vmin, vmax):
            pad = max(0.5 * bin_width_mhz, 1.0e-3)
            half_width = max(bin_width_mhz, pad)
            bin_edges = np.asarray([-half_width, 0.0, half_width], dtype=float)
        else:
            start = bin_width_mhz * (np.floor(vmin / bin_width_mhz - 0.5) + 0.5)
            stop = bin_width_mhz * (np.ceil(vmax / bin_width_mhz - 0.5) + 0.5)
            if np.isclose(start, stop):
                stop = start + bin_width_mhz
            bin_edges = np.arange(start, stop + bin_width_mhz * 1.0001, bin_width_mhz, dtype=float)
            if bin_edges.size < 2:
                bin_edges = np.asarray([start, start + bin_width_mhz], dtype=float)

        pair_labels = list(data["pair_labels"])
        values_by_pair: list[np.ndarray] = []
        plotted_pair_labels: list[str] = []
        for pair_idx, pair_label in enumerate(pair_labels):
            pair_values = np.asarray(
                [
                    float(record["rel_delta_diff"])
                    for record in data["records"]
                    if int(record["pair_idx"]) == pair_idx and np.isfinite(float(record["rel_delta_diff"]))
                ],
                dtype=float,
            )
            if pair_values.size == 0:
                continue
            values_by_pair.append(pair_values)
            plotted_pair_labels.append(pair_label)

        colors = plt.cm.rainbow(np.linspace(0.0, 1.0, max(len(values_by_pair), 1)))
        counts_list, bins, _patches = ax.hist(
            values_by_pair,
            bins=bin_edges,
            stacked=True,
            color=colors[: len(values_by_pair)],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
            label=plotted_pair_labels,
        )
        counts = np.asarray(counts_list[-1], dtype=float) if len(counts_list) else np.asarray([], dtype=float)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_xlabel("(df2 - df1) / f1")
        ax.set_ylabel("Count")
        ax.set_title("Histogram of (df2-df1)/f1 for Nearby Resonator Pairs")

        fit_mode = (
            str(self.res_pair_dfdiff_hist_fit_mode_var.get())
            if self.res_pair_dfdiff_hist_fit_mode_var is not None
            else "gaussian"
        ).strip().lower()
        x_fit = np.linspace(float(bins[0]), float(bins[-1]), 600)
        fit_summary = " Fit unavailable."
        fit_label = None
        y_fit = None
        fit_dist = None

        try:
            if fit_mode == "gennorm":
                beta_hat, mu_hat, alpha_hat = stats.gennorm.fit(values_mhz)
                if np.isfinite(beta_hat) and np.isfinite(mu_hat) and np.isfinite(alpha_hat) and beta_hat > 0.0 and alpha_hat > 0.0:
                    pdf = stats.gennorm.pdf(x_fit, beta_hat, loc=mu_hat, scale=alpha_hat)
                    y_fit = values_mhz.size * bin_width_mhz * pdf
                    fit_label = f"Gen. normal fit: mu={mu_hat:.3g}, alpha={alpha_hat:.3g}, beta={beta_hat:.3g}"
                    fit_summary = f" Gen. normal fit: mu={mu_hat:.3g}, alpha={alpha_hat:.3g}, beta={beta_hat:.3g}."
                    fit_dist = stats.gennorm(beta_hat, loc=mu_hat, scale=alpha_hat)
            else:
                mu_hat, sigma_hat = stats.norm.fit(values_mhz)
                if np.isfinite(mu_hat) and np.isfinite(sigma_hat) and sigma_hat > 0.0:
                    pdf = stats.norm.pdf(x_fit, loc=mu_hat, scale=sigma_hat)
                    y_fit = values_mhz.size * bin_width_mhz * pdf
                    fit_label = f"Gaussian fit: mu={mu_hat:.3g}, sigma={sigma_hat:.3g}"
                    fit_summary = f" Gaussian fit: mu={mu_hat:.3g}, sigma={sigma_hat:.3g}."
                    fit_dist = stats.norm(loc=mu_hat, scale=sigma_hat)
        except Exception:
            y_fit = None
            fit_dist = None

        if y_fit is not None and fit_label is not None:
            ax.plot(
                x_fit,
                y_fit,
                color="black",
                linewidth=2.0,
                linestyle="-",
                label=fit_label,
                zorder=5,
            )
        if plotted_pair_labels and len(plotted_pair_labels) <= 12:
            ax.legend(loc="best", fontsize=8, title="Consecutive tests")
        elif y_fit is not None:
            ax.legend(loc="best", fontsize=8)

        if prior_xlim is not None:
            ax.set_xlim(prior_xlim)
        else:
            ax.set_xlim(float(bins[0]), float(bins[-1]))

        y_candidates = [0.0]
        if counts.size:
            y_candidates.append(float(np.max(counts)))
        if y_fit is not None:
            fit_max = np.asarray(y_fit, dtype=float)
            fit_max = fit_max[np.isfinite(fit_max)]
            if fit_max.size:
                y_candidates.append(float(np.max(fit_max)))
        y_top = max(y_candidates)
        y_pad = 1.0 if y_top <= 0.0 else 0.08 * y_top
        ax.set_ylim(0.0, y_top + y_pad)

        start_rel_values = np.asarray(
            [float(record["start_rel_sep"]) for record in data["records"]],
            dtype=float,
        )
        end_rel_values = np.asarray(
            [float(record["end_rel_sep"]) for record in data["records"]],
            dtype=float,
        )
        valid_prob_mask = np.isfinite(start_rel_values) & np.isfinite(end_rel_values)
        start_rel_values = start_rel_values[valid_prob_mask]
        end_rel_values = end_rel_values[valid_prob_mask]
        if start_rel_values.size:
            prob_bin_width = bin_width_mhz
            prob_start = 0.0
            prob_stop = 0.001
            prob_edges = np.arange(
                prob_start,
                prob_stop + prob_bin_width * 1.0001,
                prob_bin_width,
                dtype=float,
            )
            if prob_edges.size < 2:
                prob_edges = np.asarray([prob_start, prob_start + prob_bin_width], dtype=float)

            prob_x: list[float] = []
            prob_y: list[float] = []
            prob_n: list[int] = []
            for lo, hi in zip(prob_edges[:-1], prob_edges[1:]):
                if hi <= lo:
                    continue
                if hi == prob_edges[-1]:
                    mask = (start_rel_values >= lo) & (start_rel_values <= hi)
                else:
                    mask = (start_rel_values >= lo) & (start_rel_values < hi)
                if not np.any(mask):
                    continue
                total = int(np.count_nonzero(mask))
                success = int(np.count_nonzero(np.abs(end_rel_values[mask]) <= capture_threshold))
                prob_x.append(0.5 * (lo + hi))
                prob_y.append(success / total if total > 0 else np.nan)
                prob_n.append(total)

            if prob_x:
                if fit_dist is not None:
                    x_model = np.linspace(0.0, 0.001, 500)
                    lower = -capture_threshold - x_model
                    upper = capture_threshold - x_model
                    y_model = fit_dist.cdf(upper) - fit_dist.cdf(lower)
                    ax_prob.fill_between(
                        x_model,
                        0.0,
                        y_model,
                        color="tab:blue",
                        alpha=0.2,
                        zorder=1,
                    )
                    ax_prob.plot(
                        x_model,
                        y_model,
                        color="black",
                        linewidth=2.0,
                        linestyle="-",
                        label="Model implied",
                        zorder=2,
                    )
                else:
                    ax_prob.text(
                        0.5,
                        0.5,
                        "Model fit unavailable.",
                        ha="center",
                        va="center",
                        transform=ax_prob.transAxes,
                    )
            else:
                ax_prob.text(
                    0.5,
                    0.5,
                    "No populated start-separation bins.",
                    ha="center",
                    va="center",
                    transform=ax_prob.transAxes,
                )
        else:
            ax_prob.text(
                0.5,
                0.5,
                "No finite start/end separations were available.",
                ha="center",
                va="center",
                transform=ax_prob.transAxes,
            )

        ax_prob.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--")
        ax_prob.axhline(1.0, color="0.5", linewidth=0.8, linestyle="--")
        ax_prob.grid(True, alpha=0.3)
        ax_prob.set_xlim(0.0, 0.001)
        ax_prob.set_ylim(0.0, 1.0)
        ax_prob.margins(x=0.0, y=0.0)
        ax_prob.set_xlabel("Starting relative separation (f2-f1) / f1")
        ax_prob.set_ylabel("Collision probability")
        ax_prob.set_title(f"P(|final separation| <= {capture_threshold:.3g}) vs starting separation")
        if fit_dist is not None:
            ax_prob.legend(loc="best", fontsize=8)

        if fit_dist is not None:
            pair_count = num_resonators * (num_resonators - 1) / 2.0
            rng = np.random.default_rng(12345)
            idx = np.arange(num_resonators, dtype=float) - 0.5 * (num_resonators - 1)
            base_freqs_hz = center_ghz * 1.0e9 * np.power(1.0 + dfrel_nominal, idx)
            jitter_hz = rng.normal(0.0, freq_jitter_hz, size=num_resonators)
            grid_hz = np.sort(base_freqs_hz + jitter_hz)
            ii, jj = np.triu_indices(num_resonators, k=1)
            f_lo = grid_hz[ii]
            f_hi = grid_hz[jj]
            valid_pairs = f_lo > 0.0
            rel_sep_samples = (f_hi[valid_pairs] - f_lo[valid_pairs]) / f_lo[valid_pairs]
            f_min_ghz = float(np.min(grid_hz)) / 1.0e9
            f_max_ghz = float(np.max(grid_hz)) / 1.0e9
            dist_label = (
                f"center {center_ghz:.3g} GHz, df/f {dfrel_nominal:.3g}"
                f", Gaussian offset std {freq_jitter_khz:.3g} kHz"
                f" | span {f_min_ghz:.3g}-{f_max_ghz:.3g} GHz"
            )
            if rel_sep_samples.size:
                initial_pair_probs = (rel_sep_samples <= capture_threshold).astype(float)
                lambda_initial = float(pair_count * np.mean(initial_pair_probs))
                cdf_x_max = int(max(1, num_resonators))
                x_counts = np.arange(1, cdf_x_max + 1, dtype=int)
                y_cdf_initial = stats.poisson.sf(x_counts - 1, lambda_initial)
                ax_cdf.plot(
                    x_counts,
                    y_cdf_initial,
                    color="tab:gray",
                    linewidth=1.8,
                    linestyle="--",
                    label=f"Initial model, E[K]={lambda_initial:.3g}",
                )
                step_colors = ["tab:purple", "tab:blue", "tab:green", "tab:orange", "tab:red"]
                lambda_by_step: dict[int, float] = {}
                delta_sample_count = 40000
                base_delta_samples = np.asarray(
                    fit_dist.rvs(size=(5, delta_sample_count), random_state=rng),
                    dtype=float,
                )
                for step_count, color in zip(range(1, 6), step_colors):
                    summed_delta = np.sum(base_delta_samples[:step_count, :], axis=0)
                    summed_delta = np.sort(summed_delta[np.isfinite(summed_delta)])
                    if summed_delta.size == 0:
                        continue
                    upper = capture_threshold - rel_sep_samples
                    lower = -capture_threshold - rel_sep_samples
                    upper_idx = np.searchsorted(summed_delta, upper, side="right")
                    lower_idx = np.searchsorted(summed_delta, lower, side="left")
                    model_pair_probs = (upper_idx - lower_idx) / float(summed_delta.size)
                    model_pair_probs = np.clip(np.asarray(model_pair_probs, dtype=float), 0.0, 1.0)
                    lambda_step = float(pair_count * np.mean(model_pair_probs))
                    lambda_by_step[step_count] = lambda_step
                    y_cdf = stats.poisson.sf(x_counts - 1, lambda_step)
                    ax_cdf.plot(
                        x_counts,
                        y_cdf,
                        color=color,
                        linewidth=2.0,
                        label=f"{step_count} step{'s' if step_count != 1 else ''}, E[K]={lambda_step:.3g}",
                    )
                ax_cdf.set_xlim(0, cdf_x_max)
                ax_cdf.set_xscale("log")
                ax_cdf.set_xlim(1, cdf_x_max)
                ax_cdf.set_ylim(0.0, 1.0)
                ax_cdf.set_xlabel("x collisions")
                ax_cdf.set_ylabel("P(K >= x)")
                ax_cdf.set_title(
                    f"Collision Count Exceedance | N={num_resonators}, {dist_label}"
                )
                ax_cdf.grid(True, alpha=0.3)
                ax_cdf.legend(loc="best", fontsize=8)
                ax_cdf.text(
                    0.01,
                    0.02,
                    "Note: E[K] from pair Monte Carlo/exact pair set; P(K>=x) from Poisson approximation.",
                    transform=ax_cdf.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=8,
                    color="0.35",
                )
                lambda_expected = float(lambda_by_step.get(1, np.nan))
            else:
                lambda_initial = np.nan
                lambda_expected = np.nan
                ax_cdf.text(
                    0.5,
                    0.5,
                    "No valid random pair samples were available.",
                    ha="center",
                    va="center",
                    transform=ax_cdf.transAxes,
                )
                ax_cdf.set_axis_off()
        else:
            lambda_initial = np.nan
            lambda_expected = np.nan
            ax_cdf.text(
                0.5,
                0.5,
                "Fit model unavailable for collision-count estimate.",
                ha="center",
                va="center",
                transform=ax_cdf.transAxes,
            )
            ax_cdf.set_axis_off()

        self.res_pair_dfdiff_hist_figure.tight_layout()
        if self.res_pair_dfdiff_hist_status_var is not None:
            pileup_text = (
                f" Expected initial collisions: {lambda_initial:.3g}. Expected post-shift collisions: {lambda_expected:.3g}. Layout: {dist_label}."
                if np.isfinite(lambda_expected)
                else " Expected collisions unavailable."
            )
            self.res_pair_dfdiff_hist_status_var.set(
                f"Showing {int(values_mhz.size)} (df2-df1)/f1 value(s) from {len(data['records'])} resonator pair(s) across {len(data['pair_labels'])} consecutive test pair(s). Max |f2-f1|={max_sep_mhz:.3g} MHz, bin width={bin_width_mhz:.3g}, collision +/-={capture_threshold:.3g}, peak count={int(np.max(counts)) if counts.size else 0}.{fit_summary}{pileup_text}"
            )
        self.res_pair_dfdiff_hist_canvas.draw_idle()
