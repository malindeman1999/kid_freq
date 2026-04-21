from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

from analysis_gui_support.analysis_models import VNAScan

class ResonatorNeighborDataMixin:
    @staticmethod
    def _resonator_shift_test_sort_stamp(scan: VNAScan) -> str:
        text = str(getattr(scan, "file_timestamp", "") or "").strip()
        if text:
            return text
        return str(getattr(scan, "loaded_at", "") or "").strip()



    @staticmethod
    def _resonator_shift_test_date_label(scan: VNAScan) -> str:
        stamp = ResonatorNeighborDataMixin._resonator_shift_test_sort_stamp(scan)
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
