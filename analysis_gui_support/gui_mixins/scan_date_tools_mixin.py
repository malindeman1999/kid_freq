from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import messagebox

from analysis_gui_support.analysis_models import _make_event

class ScanDateToolsMixin:
    @staticmethod
    def _scan_sort_stamp(scan: VNAScan) -> str:
        file_stamp = str(getattr(scan, "file_timestamp", "") or "").strip()
        if file_stamp:
            return file_stamp
        return str(getattr(scan, "loaded_at", "") or "").strip()


    @staticmethod
    def _scan_sort_date_label(scan: VNAScan) -> str:
        stamp = ScanDateToolsMixin._scan_sort_stamp(scan)
        if not stamp:
            return "unknown date"
        return stamp.split("T", 1)[0]


    @staticmethod
    def _scan_sort_key(scan: VNAScan) -> tuple[int, object, str]:
        stamp = ScanDateToolsMixin._scan_sort_stamp(scan)
        if scan.plot_group is None:
            group_key = (1, 0)
        else:
            group_key = (0, int(scan.plot_group))
        name_key = Path(str(scan.filename)).name.lower()
        if stamp:
            try:
                return (0, datetime.fromisoformat(stamp), group_key, name_key)
            except Exception:
                date_part = stamp.split("T", 1)[0]
                try:
                    return (0, datetime.fromisoformat(date_part), group_key, name_key)
                except Exception:
                    return (1, stamp.lower(), group_key, name_key)
        return (2, "", group_key, name_key)


    @staticmethod
    def _date_from_source_dir_name(source_dir: str) -> Optional[str]:
        source_text = str(source_dir).strip()
        if not source_text:
            return None
        candidates = [Path(source_text).name, source_text]
        for text in candidates:
            for match in re.finditer(r"(?<!\d)(\d{8})(?!\d)", text):
                token = match.group(1)
                try:
                    dt = datetime.strptime(token, "%Y%m%d")
                except Exception:
                    continue
                return dt.date().isoformat()
            for match in re.finditer(r"(?<!\d)(\d{6})(?=_|\s|$)", text):
                token = match.group(1)
                try:
                    dt = datetime.strptime(token, "%y%m%d")
                except Exception:
                    continue
                return dt.date().isoformat()
        return None


    @staticmethod
    def _date_from_filename_8digit(filename: str) -> Optional[str]:
        name_text = Path(str(filename)).name.strip()
        if not name_text:
            return None
        for match in re.finditer(r"(?<!\d)(\d{8})(?!\d)", name_text):
            token = match.group(1)
            try:
                dt = datetime.strptime(token, "%Y%m%d")
            except Exception:
                continue
            return dt.date().isoformat()
        return None


    @staticmethod
    def _replace_iso_date_fixed_1pm(new_date_iso: str) -> str:
        return f"{new_date_iso}T13:00:00"


    def _update_selected_vna_dates(self, mode: str) -> None:
        scans = self._selected_scans()
        if not scans:
            messagebox.showwarning(
                "No selection",
                "No scans selected for analysis.\nUse 'Select Scans for Analysis' first.",
            )
            return

        if mode == "filename":
            parse_date = lambda scan: self._date_from_filename_8digit(getattr(scan, "filename", ""))
            skip_text = (
                lambda scan: f"{self._scan_file_two_level_context(scan)}: "
                f"no YYYYMMDD date token found in filename {Path(str(getattr(scan, 'filename', ''))).name}"
            )
            confirm_title = "Update VNA Dates From Filename"
            confirm_message = (
                "The following selected VNA scans will have their file date updated to a date parsed from the filename "
                "(first YYYYMMDD token). This updated date will be used in plots."
            )
            no_updates_message = "No selected scans need a date update from filename."
            per_scan_event = "update_vna_file_timestamp_from_filename"
            dataset_event = "update_selected_vna_dates_from_filename"
            log_mode_text = "filename date"
            info_mode_text = "filename"
        else:
            parse_date = lambda scan: self._date_from_source_dir_name(getattr(scan, "source_dir", ""))
            skip_text = (
                lambda scan: f"{self._scan_file_two_level_context(scan)}: "
                f"no YYYYMMDD or YYMMDD_ / YYMMDD<space> date token found in {scan.source_dir}"
            )
            confirm_title = "Update VNA Dates From Source Directory"
            confirm_message = (
                "The following selected VNA scans will have their file date updated from the filesystem date to a date "
                "parsed from source_dir. This updated date will be used in plots."
            )
            no_updates_message = "No selected scans need a date update from source_dir."
            per_scan_event = "update_vna_file_timestamp_from_source_dir"
            dataset_event = "update_selected_vna_dates_from_source_dir"
            log_mode_text = "source_dir date"
            info_mode_text = "source_dir"

        proposed_updates: list[tuple[VNAScan, str, str, str]] = []
        skipped: list[str] = []
        for scan in scans:
            new_date_iso = parse_date(scan)
            if new_date_iso is None:
                skipped.append(skip_text(scan))
                continue
            old_timestamp = str(getattr(scan, "file_timestamp", "") or "").strip()
            new_timestamp = self._replace_iso_date_fixed_1pm(new_date_iso)
            if new_timestamp == old_timestamp:
                continue
            proposed_updates.append((scan, old_timestamp, new_timestamp, new_date_iso))

        if not proposed_updates:
            message = no_updates_message
            if skipped:
                message += "\n\nSkipped:\n" + "\n".join(skipped[:10])
            messagebox.showwarning("No updates", message)
            return

        preview_lines = [
            f"{self._scan_file_two_level_context(scan)}: {old if old else 'unknown'} -> {new}"
            for scan, old, new, _date_iso in proposed_updates
        ]
        if skipped:
            preview_lines.append("")
            preview_lines.append("Skipped:")
            preview_lines.extend(skipped[:10])
            if len(skipped) > 10:
                preview_lines.append(f"... and {len(skipped) - 10} more")

        approved = self._confirm_bulk_text_changes(
            confirm_title,
            confirm_message,
            preview_lines,
        )
        if not approved:
            return

        changed_count = 0
        for scan, old_timestamp, new_timestamp, new_date_iso in proposed_updates:
            scan.file_timestamp = new_timestamp
            scan.processing_history.append(
                _make_event(
                    per_scan_event,
                    {
                        "filename": scan.filename,
                        "source_dir": scan.source_dir,
                        "old_file_timestamp": old_timestamp,
                        "new_file_timestamp": new_timestamp,
                        "parsed_date": new_date_iso,
                    },
                )
            )
            changed_count += 1

        self.dataset.processing_history.append(
            _make_event(
                dataset_event,
                {
                    "selected_count": len(scans),
                    "updated_count": changed_count,
                    "skipped_count": len(skipped),
                },
            )
        )
        self._mark_dirty()
        self._refresh_status()
        self._autosave_dataset()
        self._log(
            f"Updated {changed_count} selected VNA file timestamp(s) from {log_mode_text}; skipped {len(skipped)}."
        )
        messagebox.showinfo(
            "Dates updated",
            f"Updated {changed_count} selected scan date(s) from {info_mode_text}."
            + (f"\nSkipped {len(skipped)} scan(s)." if skipped else ""),
        )


    def update_selected_vna_dates_from_source_dir(self) -> None:
        self._update_selected_vna_dates(mode="dir")


    def update_selected_vna_dates_from_path(self) -> None:
        mode = str(self.update_dates_mode_var.get() or "dir").strip().lower()
        if mode not in {"dir", "filename"}:
            mode = "dir"
        self._update_selected_vna_dates(mode=mode)


    def open_update_dates_dialog(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Update Dates From Path")
        dialog.geometry("420x190")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(
            dialog,
            text="Choose where to parse scan dates from:",
            anchor="w",
            justify="left",
        ).pack(anchor="w", padx=12, pady=(12, 8))

        mode_var = tk.StringVar(value=str(self.update_dates_mode_var.get() or "dir"))
        tk.Radiobutton(
            dialog,
            text="Directory (default behavior)",
            value="dir",
            variable=mode_var,
            anchor="w",
            justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 4))
        tk.Radiobutton(
            dialog,
            text="Filename (first YYYYMMDD token)",
            value="filename",
            variable=mode_var,
            anchor="w",
            justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 8))

        button_row = tk.Frame(dialog)
        button_row.pack(fill="x", padx=12, pady=(4, 12))

        def _run() -> None:
            chosen = str(mode_var.get() or "dir").strip().lower()
            if chosen not in {"dir", "filename"}:
                chosen = "dir"
            self.update_dates_mode_var.set(chosen)
            dialog.destroy()
            self._update_selected_vna_dates(chosen)

        tk.Button(button_row, text="Cancel", width=10, command=dialog.destroy).pack(side="right")
        tk.Button(button_row, text="Update", width=10, command=_run).pack(side="right", padx=(0, 8))


    def reorder_vna_scans_by_date(self) -> None:
        scans = list(self.dataset.vna_scans)
        if not scans:
            messagebox.showwarning("No data", "No VNA scans are loaded in this dataset.")
            return

        reordered = sorted(scans, key=self._scan_sort_key)
        if [id(scan) for scan in reordered] == [id(scan) for scan in scans]:
            messagebox.showinfo("Already ordered", "Scans are already ordered by date.")
            return

        preview_lines = []
        for idx, scan in enumerate(reordered):
            group_text = f"group {int(scan.plot_group)}" if scan.plot_group is not None else "no group"
            preview_lines.append(
                f"{idx:03d} | {self._scan_file_two_level_context(scan)} | {group_text} | {self._scan_sort_date_label(scan)}"
            )

        approved = self._confirm_bulk_text_changes(
            "Reorder VNA Scans By Date",
            "The scans below are in the proposed date-sorted order. Approve to reorder dataset scan order.",
            preview_lines,
        )
        if not approved:
            return

        selected_set = set(self.dataset.selected_scan_keys)
        self.dataset.vna_scans = reordered
        self.dataset.selected_scan_keys = [
            self._scan_key(scan) for scan in self.dataset.vna_scans if self._scan_key(scan) in selected_set
        ]
        self.dataset.processing_history.append(
            _make_event(
                "reorder_vna_scans_by_date",
                {
                    "scan_count": len(self.dataset.vna_scans),
                },
            )
        )
        self._mark_dirty()
        self._refresh_status()
        self._autosave_dataset()
        self._log(f"Reordered {len(self.dataset.vna_scans)} VNA scan(s) by date.")
        messagebox.showinfo(
            "Scans reordered",
            f"Reordered {len(self.dataset.vna_scans)} scan(s) by date. Dataset saved automatically.",
        )
