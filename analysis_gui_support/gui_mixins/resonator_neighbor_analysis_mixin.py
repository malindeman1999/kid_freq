from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from matplotlib import colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from scipy import stats
from tkinter import messagebox

from analysis_gui_support.analysis_io import _dataset_dir
from analysis_gui_support.analysis_models import VNAScan, _make_event

class ResonatorNeighborAnalysisMixin:
    @staticmethod
    def _resonator_shift_test_sort_stamp(scan: VNAScan) -> str:
        text = str(getattr(scan, "file_timestamp", "") or "").strip()
        if text:
            return text
        return str(getattr(scan, "loaded_at", "") or "").strip()


    @staticmethod
    def _resonator_shift_test_date_label(scan: VNAScan) -> str:
        stamp = ResonatorNeighborAnalysisMixin._resonator_shift_test_sort_stamp(scan)
        if stamp:
            return stamp.split("T", 1)[0]
        return "unknown date"


    @staticmethod
    def _resonator_shift_parse_timestamp(text: object) -> Optional[datetime]:
        stamp = str(text or "").strip()
        if not stamp:
            return None
        try:
            return datetime.fromisoformat(stamp)
        except Exception:
            pass
        for fmt in ("%Y-%m-%d", "%Y%m%d_%H%M%S", "%Y%m%d"):
            try:
                return datetime.strptime(stamp, fmt)
            except Exception:
                continue
        return None


    def _resonator_shift_test_units(self) -> list[dict]:
        scans = self._selected_scans()
        if not scans:
            raise ValueError("No scans are selected for analysis.")

        units_by_key: dict[tuple[str, object], dict] = {}
        ordered_keys: list[tuple[str, object]] = []
        for scan_index, scan in enumerate(scans):
            payload = scan.candidate_resonators.get("sheet_resonances")
            if not isinstance(payload, dict):
                continue
            assignments = payload.get("assignments")
            if not isinstance(assignments, dict):
                continue
            if scan.plot_group is not None:
                unit_key: tuple[str, object] = ("group", int(scan.plot_group))
                detail_label = f"Group {int(scan.plot_group)}"
            else:
                unit_key = ("scan", self._scan_key(scan))
                detail_label = Path(scan.filename).name
            unit = units_by_key.get(unit_key)
            if unit is None:
                sort_stamp = self._resonator_shift_test_sort_stamp(scan)
                unit = {
                    "key": unit_key,
                    "sort_stamp": sort_stamp,
                    "timestamp_dt": self._resonator_shift_parse_timestamp(sort_stamp),
                    "order_index": scan_index,
                    "date_label": self._resonator_shift_test_date_label(scan),
                    "detail_label": detail_label,
                    "resonator_lists": {},
                }
                units_by_key[unit_key] = unit
                ordered_keys.append(unit_key)
            else:
                sort_stamp = self._resonator_shift_test_sort_stamp(scan)
                if sort_stamp and (not unit["sort_stamp"] or sort_stamp < unit["sort_stamp"]):
                    unit["sort_stamp"] = sort_stamp
                    unit["timestamp_dt"] = self._resonator_shift_parse_timestamp(sort_stamp)
                    unit["date_label"] = self._resonator_shift_test_date_label(scan)

            freq = np.asarray(scan.freq, dtype=float)
            if freq.size == 0:
                continue
            freq_min = float(np.min(freq))
            freq_max = float(np.max(freq))
            resonator_lists = unit["resonator_lists"]
            for resonator_number, record in assignments.items():
                if not isinstance(record, dict):
                    continue
                try:
                    target_hz = float(record.get("frequency_hz"))
                except Exception:
                    continue
                if not np.isfinite(target_hz) or not (freq_min <= target_hz <= freq_max):
                    continue
                resonator_label = str(resonator_number).strip()
                if not resonator_label:
                    continue
                resonator_lists.setdefault(resonator_label, []).append(target_hz)

        units: list[dict] = []
        for unit_key in ordered_keys:
            unit = units_by_key[unit_key]
            resonators: dict[str, float] = {}
            for resonator_label, values in unit["resonator_lists"].items():
                arr = np.asarray(values, dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size:
                    resonators[resonator_label] = float(np.mean(arr))
            if not resonators:
                continue
            units.append(
                {
                    "key": unit["key"],
                    "sort_stamp": str(unit["sort_stamp"]),
                    "timestamp_dt": unit.get("timestamp_dt"),
                    "order_index": int(unit["order_index"]),
                    "date_label": str(unit["date_label"]),
                    "detail_label": str(unit["detail_label"]),
                    "label": f"{unit['date_label']} | {unit['detail_label']}",
                    "resonators": resonators,
                }
            )

        units.sort(key=lambda item: (str(item["sort_stamp"]), int(item["order_index"])))
        return units


    @staticmethod
    def _resonator_neighbor_parse_initial_date(text: str) -> Optional[datetime]:
        date_text = str(text).strip()
        if not date_text:
            return None
        try:
            return datetime.strptime(date_text, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError("Initial date must use YYYY-MM-DD.") from exc


    def _resonator_neighbor_dfrel_data(
        self,
        max_neighbor_sep_rel: float,
        initial_date_text: str = "",
    ) -> dict:
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
                values_by_resonator.setdefault(resonator_label, []).append(float(freq_hz))

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
        if len(ordered_labels) < 2:
            raise ValueError("Need at least two marked resonators to build neighboring pairs.")

        threshold_rel = max(0.0, float(max_neighbor_sep_rel))
        pair_series: list[dict] = []
        for idx in range(len(ordered_labels) - 1):
            low_label = ordered_labels[idx]
            high_label = ordered_labels[idx + 1]
            low_mean_hz = float(mean_freq_by_resonator[low_label])
            high_mean_hz = float(mean_freq_by_resonator[high_label])
            pair_mean_hz = 0.5 * (low_mean_hz + high_mean_hz)
            pair_sep_hz = high_mean_hz - low_mean_hz
            pair_sep_rel = pair_sep_hz / pair_mean_hz if pair_mean_hz != 0.0 else np.nan
            if not np.isfinite(pair_sep_rel) or pair_sep_rel < 0.0 or pair_sep_rel > threshold_rel:
                continue
            points: list[dict] = []
            for test in tests:
                elapsed_days = float(test.get("elapsed_days", np.nan))
                if not np.isfinite(elapsed_days):
                    continue
                test_resonators = test["resonators"]
                if low_label not in test_resonators or high_label not in test_resonators:
                    continue
                low_freq_hz = float(test_resonators[low_label])
                high_freq_hz = float(test_resonators[high_label])
                df_hz = high_freq_hz - low_freq_hz
                if not np.isfinite(df_hz) or pair_mean_hz == 0.0:
                    continue
                points.append(
                    {
                        "elapsed_days": elapsed_days,
                        "df_over_f": float(df_hz / pair_mean_hz),
                        "df_hz": df_hz,
                        "low_freq_hz": low_freq_hz,
                        "high_freq_hz": high_freq_hz,
                        "test_label": test["label"],
                    }
                )
            if len(points) < 2:
                continue
            points.sort(key=lambda item: float(item["elapsed_days"]))
            pair_series.append(
                {
                    "low_label": low_label,
                    "high_label": high_label,
                    "label": f"{low_label}-{high_label}",
                    "mean_freq_hz": pair_mean_hz,
                    "mean_sep_hz": pair_sep_hz,
                    "mean_sep_rel": float(pair_sep_rel),
                    "points": points,
                }
            )

        if not pair_series:
            raise ValueError(
                "No adjacent resonator pairs met the relative df/f threshold with at least two dated measurements."
            )

        pair_series.sort(
            key=lambda item: (float(item["mean_freq_hz"]), self._resonator_sort_key(str(item["label"])))
        )
        mean_pair_freqs_hz = np.asarray([float(item["mean_freq_hz"]) for item in pair_series], dtype=float)
        return {
            "tests": tests,
            "pair_series": pair_series,
            "mean_pair_freqs_hz": mean_pair_freqs_hz,
            "threshold_rel": float(max_neighbor_sep_rel),
            "elapsed_time_origin": base_time,
            "elapsed_time_origin_source": "initial_date" if initial_time is not None else "first_dataset",
        }


    @staticmethod
    def _resonator_neighbor_pair_colors(data: dict) -> tuple[mcolors.Normalize, object]:
        mean_pair_freqs_hz = np.asarray(data["mean_pair_freqs_hz"], dtype=float)
        vmin = float(np.min(mean_pair_freqs_hz))
        vmax = float(np.max(mean_pair_freqs_hz))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            raise ValueError("Could not determine pair mean frequencies for coloring.")
        if vmax <= vmin:
            vmax = vmin + 1.0
        return mcolors.Normalize(vmin=vmin, vmax=vmax), plt.cm.get_cmap("rainbow_r")


    def _resonator_neighbor_scan_overlay_state(self, threshold_rel: float, initial_date_text: str = "") -> dict:
        data = self._resonator_neighbor_dfrel_data(threshold_rel, initial_date_text=initial_date_text)
        norm, cmap = self._resonator_neighbor_pair_colors(data)
        pair_series = data["pair_series"]

        pair_color_by_label: dict[str, object] = {}
        resonator_color_lists: dict[str, list[np.ndarray]] = {}
        for pair in pair_series:
            color = cmap(norm(float(pair["mean_freq_hz"])))
            pair["color"] = color
            pair_color_by_label[str(pair["label"])] = color
            for resonator_label in (str(pair["low_label"]), str(pair["high_label"])):
                resonator_color_lists.setdefault(resonator_label, []).append(np.asarray(color[:3], dtype=float))

        resonator_colors: dict[str, tuple[float, float, float, float]] = {}
        for resonator_label, colors in resonator_color_lists.items():
            if not colors:
                continue
            mean_rgb = np.mean(np.vstack(colors), axis=0)
            resonator_colors[resonator_label] = (
                float(np.clip(mean_rgb[0], 0.0, 1.0)),
                float(np.clip(mean_rgb[1], 0.0, 1.0)),
                float(np.clip(mean_rgb[2], 0.0, 1.0)),
                1.0,
            )

        return {
            "data": data,
            "norm": norm,
            "cmap": cmap,
            "pair_color_by_label": pair_color_by_label,
            "resonator_colors": resonator_colors,
        }


    @staticmethod
    def _resonator_neighbor_summary_by_time(pair_series: list[dict]) -> list[dict]:
        values_by_time: dict[float, list[float]] = {}
        for pair in pair_series:
            for point in pair["points"]:
                elapsed_days = float(point["elapsed_days"])
                df_over_f = float(point.get("plot_df_over_f", point["df_over_f"]))
                if not np.isfinite(elapsed_days) or not np.isfinite(df_over_f):
                    continue
                values_by_time.setdefault(elapsed_days, []).append(df_over_f)

        summary: list[dict] = []
        for elapsed_days in sorted(values_by_time):
            arr = np.asarray(values_by_time[elapsed_days], dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            mean_value = float(np.mean(arr))
            std_value = float(np.std(arr))
            q1, median, q3 = np.percentile(arr, [25.0, 50.0, 75.0])
            iqr = float(q3 - q1)
            lower_bound = float(q1 - 1.5 * iqr)
            upper_bound = float(q3 + 1.5 * iqr)
            whisker_low = float(np.min(arr[arr >= lower_bound])) if np.any(arr >= lower_bound) else float(np.min(arr))
            whisker_high = float(np.max(arr[arr <= upper_bound])) if np.any(arr <= upper_bound) else float(np.max(arr))
            outliers = arr[(arr < whisker_low) | (arr > whisker_high)]
            summary.append(
                {
                    "elapsed_days": float(elapsed_days),
                    "count": int(arr.size),
                    "mean": mean_value,
                    "std": std_value,
                    "lower": float(mean_value - std_value),
                    "upper": float(mean_value + std_value),
                    "q1": float(q1),
                    "median": float(median),
                    "q3": float(q3),
                    "whisker_low": whisker_low,
                    "whisker_high": whisker_high,
                    "outliers": np.asarray(outliers, dtype=float),
                }
            )
        return summary


    @staticmethod
    def _resonator_neighbor_plot_series(pair_series: list[dict], mode: str) -> list[dict]:
        mode_key = str(mode).strip().lower()
        plotted: list[dict] = []
        for pair in pair_series:
            raw_points = pair.get("points", [])
            if not raw_points:
                continue
            baseline = float(raw_points[0]["df_over_f"])
            points: list[dict] = []
            for point in raw_points:
                y_value = float(point["df_over_f"])
                if mode_key == "change":
                    y_value -= baseline
                updated = dict(point)
                updated["plot_df_over_f"] = float(y_value)
                points.append(updated)
            updated_pair = dict(pair)
            updated_pair["points"] = points
            plotted.append(updated_pair)
        return plotted


    @staticmethod
    def _resonator_neighbor_drift_rate_summary(pair_series: list[dict]) -> list[dict]:
        values_by_time: dict[float, list[float]] = {}
        for pair in pair_series:
            points = pair.get("points", [])
            if len(points) < 2:
                continue
            for prev_point, point in zip(points[:-1], points[1:]):
                elapsed_prev = float(prev_point["elapsed_days"])
                elapsed_days = float(point["elapsed_days"])
                y_prev = float(prev_point.get("plot_df_over_f", prev_point["df_over_f"]))
                y_value = float(point.get("plot_df_over_f", point["df_over_f"]))
                delta_days = elapsed_days - elapsed_prev
                if (
                    not np.isfinite(elapsed_prev)
                    or not np.isfinite(elapsed_days)
                    or not np.isfinite(y_prev)
                    or not np.isfinite(y_value)
                    or delta_days <= 0.0
                ):
                    continue
                values_by_time.setdefault(elapsed_days, []).append(float((y_value - y_prev) / delta_days))

        summary: list[dict] = []
        for elapsed_days in sorted(values_by_time):
            arr = np.asarray(values_by_time[elapsed_days], dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            mean_value = float(np.mean(arr))
            std_value = float(np.std(arr))
            summary.append(
                {
                    "elapsed_days": float(elapsed_days),
                    "count": int(arr.size),
                    "mean": mean_value,
                    "std": std_value,
                    "lower": float(mean_value - std_value),
                    "upper": float(mean_value + std_value),
                }
            )
        return summary


    @staticmethod
    def _resonator_neighbor_drift_rate_series(pair_series: list[dict]) -> list[dict]:
        drift_series: list[dict] = []
        for pair in pair_series:
            points = pair.get("points", [])
            if len(points) < 2:
                continue
            drift_points: list[dict] = []
            for prev_point, point in zip(points[:-1], points[1:]):
                elapsed_prev = float(prev_point["elapsed_days"])
                elapsed_days = float(point["elapsed_days"])
                y_prev = float(prev_point.get("plot_df_over_f", prev_point["df_over_f"]))
                y_value = float(point.get("plot_df_over_f", point["df_over_f"]))
                delta_days = elapsed_days - elapsed_prev
                if (
                    not np.isfinite(elapsed_prev)
                    or not np.isfinite(elapsed_days)
                    or not np.isfinite(y_prev)
                    or not np.isfinite(y_value)
                    or delta_days <= 0.0
                ):
                    continue
                drift_points.append(
                    {
                        "elapsed_days": float(elapsed_days),
                        "drift_rate": float((y_value - y_prev) / delta_days),
                    }
                )
            if not drift_points:
                continue
            updated_pair = dict(pair)
            updated_pair["drift_points"] = drift_points
            drift_series.append(updated_pair)
        return drift_series


    @staticmethod
    def _resonator_neighbor_interval_drift_data(data: dict) -> list[dict]:
        tests = list(data.get("tests", []))
        pair_series = list(data.get("pair_series", []))
        if len(tests) < 2 or not pair_series:
            return []

        intervals: list[dict] = []
        for prev_test, test in zip(tests[:-1], tests[1:]):
            elapsed_prev = float(prev_test.get("elapsed_days", np.nan))
            elapsed_days = float(test.get("elapsed_days", np.nan))
            delta_days = elapsed_days - elapsed_prev
            if (
                not np.isfinite(elapsed_prev)
                or not np.isfinite(elapsed_days)
                or delta_days <= 0.0
            ):
                continue
            intervals.append(
                {
                    "start_label": str(prev_test.get("label", "")),
                    "end_label": str(test.get("label", "")),
                    "start_elapsed_days": elapsed_prev,
                    "end_elapsed_days": elapsed_days,
                    "mid_elapsed_days": float(0.5 * (elapsed_prev + elapsed_days)),
                    "delta_days": float(delta_days),
                    "pair_drift_by_label": {},
                }
            )

        if not intervals:
            return []

        for pair in pair_series:
            pair_label = str(pair.get("label", ""))
            point_by_test_label = {
                str(point.get("test_label", "")): point
                for point in pair.get("points", [])
                if str(point.get("test_label", "")).strip()
            }
            for interval in intervals:
                prev_point = point_by_test_label.get(interval["start_label"])
                point = point_by_test_label.get(interval["end_label"])
                if prev_point is None or point is None:
                    continue
                y_prev = float(prev_point["df_over_f"])
                y_value = float(point["df_over_f"])
                delta_days = float(interval["delta_days"])
                if (
                    not np.isfinite(y_prev)
                    or not np.isfinite(y_value)
                    or delta_days <= 0.0
                ):
                    continue
                interval["pair_drift_by_label"][pair_label] = float((y_value - y_prev) / delta_days)
        return intervals


    @staticmethod
    def _resonator_neighbor_pair_correlation(x_values: np.ndarray, y_values: np.ndarray) -> float:
        x = np.asarray(x_values, dtype=float)
        y = np.asarray(y_values, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size == 0:
            return np.nan
        if x.size == 1:
            return 1.0 if np.allclose(x, y, rtol=0.0, atol=0.0) else np.nan
        x_centered = x - float(np.mean(x))
        y_centered = y - float(np.mean(y))
        denom = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
        if denom <= 0.0:
            return 1.0 if np.allclose(x, y, rtol=0.0, atol=0.0) else np.nan
        return float(np.dot(x_centered, y_centered) / denom)


    def _resonator_neighbor_self_correlation_summary(self, data: dict) -> dict:
        intervals = self._resonator_neighbor_interval_drift_data(data)
        if not intervals:
            raise ValueError("Need at least one consecutive dated interval to analyze pair drift correlation.")

        reference_idx = next(
            (
                idx
                for idx, interval in enumerate(intervals)
                if len(interval["pair_drift_by_label"]) >= 2
            ),
            None,
        )
        if reference_idx is None:
            raise ValueError("Need at least two neighboring pairs on one interval to compute correlation.")

        reference_interval = intervals[reference_idx]
        reference_drift = dict(reference_interval["pair_drift_by_label"])
        pair_mean_freq_by_label = {
            str(pair.get("label", "")): float(pair.get("mean_freq_hz", np.nan))
            for pair in data.get("pair_series", [])
        }
        points: list[dict] = []
        for interval_idx, interval in enumerate(intervals):
            current_drift = dict(interval["pair_drift_by_label"])
            common_labels = sorted(
                set(reference_drift).intersection(current_drift),
                key=self._resonator_sort_key,
            )
            corr_value = np.nan
            q1_value = np.nan
            q3_value = np.nan
            contribution_by_label: dict[str, float] = {}
            if common_labels:
                x = np.asarray([float(reference_drift[label]) for label in common_labels], dtype=float)
                y = np.asarray([float(current_drift[label]) for label in common_labels], dtype=float)
                corr_value = self._resonator_neighbor_pair_correlation(x, y)
                if x.size >= 2:
                    x_centered = x - float(np.mean(x))
                    y_centered = y - float(np.mean(y))
                    x_std = float(np.std(x))
                    y_std = float(np.std(y))
                    if x_std > 0.0 and y_std > 0.0:
                        contributions = (x_centered / x_std) * (y_centered / y_std)
                        for label, value in zip(common_labels, contributions):
                            contribution_by_label[str(label)] = float(value)
                        q1_value, q3_value = np.percentile(contributions, [25.0, 75.0])
                if interval_idx == reference_idx and np.isfinite(corr_value):
                    corr_value = 1.0

            drift_arr = np.asarray(list(current_drift.values()), dtype=float)
            drift_arr = drift_arr[np.isfinite(drift_arr)]
            abs_arr = np.abs(drift_arr)
            mean_abs = float(np.mean(abs_arr)) if abs_arr.size else np.nan
            std_abs = float(np.std(abs_arr)) if abs_arr.size else np.nan
            rms = float(np.sqrt(np.mean(np.square(drift_arr)))) if drift_arr.size else np.nan

            points.append(
                {
                    "interval_idx": int(interval_idx),
                    "pair_count": int(len(current_drift)),
                    "common_pair_count": int(len(common_labels)),
                    "start_elapsed_days": float(interval["start_elapsed_days"]),
                    "mid_elapsed_days": float(interval["mid_elapsed_days"]),
                    "end_elapsed_days": float(interval["end_elapsed_days"]),
                    "delta_days": float(interval["delta_days"]),
                    "corr_to_initial": float(corr_value) if np.isfinite(corr_value) else np.nan,
                    "q1_contribution": float(q1_value) if np.isfinite(q1_value) else np.nan,
                    "q3_contribution": float(q3_value) if np.isfinite(q3_value) else np.nan,
                    "contribution_by_label": contribution_by_label,
                    "mean_abs_drift_rate": mean_abs,
                    "std_abs_drift_rate": std_abs,
                    "rms_drift_rate": rms,
                }
            )

        return {
            "intervals": intervals,
            "points": points,
            "reference_interval_idx": int(reference_idx),
            "reference_interval": reference_interval,
            "pair_mean_freq_by_label": pair_mean_freq_by_label,
        }


    def _resonator_neighbor_scan_control_values(self, source: str | None = None) -> tuple[float, str]:
        source_key = str(source or self._res_neighbor_scan_source or "dfrel").strip().lower()
        if source_key == "corr":
            threshold_rel = (
                float(self.res_neighbor_corr_sep_rel_var.get())
                if self.res_neighbor_corr_sep_rel_var is not None
                else 0.004
            )
            initial_date_text = (
                str(self.res_neighbor_corr_initial_date_var.get())
                if self.res_neighbor_corr_initial_date_var is not None
                else self._dataset_res_neighbor_initial_date()
            )
            return threshold_rel, initial_date_text

        threshold_rel = (
            float(self.res_neighbor_dfrel_sep_rel_var.get())
            if self.res_neighbor_dfrel_sep_rel_var is not None
            else 0.004
        )
        initial_date_text = (
            str(self.res_neighbor_dfrel_initial_date_var.get())
            if self.res_neighbor_dfrel_initial_date_var is not None
            else self._dataset_res_neighbor_initial_date()
        )
        return threshold_rel, initial_date_text


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
