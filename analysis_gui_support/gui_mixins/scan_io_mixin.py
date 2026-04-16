from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import tkinter as tk
from tkinter import filedialog, messagebox

from analysis_gui_support.analysis_io import _load_vna_file, _load_vna_npy_mhz_db_deg, _try_load_vna_npy_pair
from analysis_gui_support.analysis_models import VNAScan, _make_event

class ScanIOMixin:
    @staticmethod
    def _scan_key(scan: VNAScan) -> str:
        return f"{scan.filename}|{scan.loaded_at}"


    def _prune_selected_scan_keys(self) -> None:
        valid_keys = {self._scan_key(scan) for scan in self.dataset.vna_scans}
        self.dataset.selected_scan_keys = [
            key for key in self.dataset.selected_scan_keys if key in valid_keys
        ]


    def _selected_scans(self) -> List[VNAScan]:
        selected_keys = set(self.dataset.selected_scan_keys)
        return [scan for scan in self.dataset.vna_scans if self._scan_key(scan) in selected_keys]


    def _choose_one_selected_scan(self) -> Optional[VNAScan]:
        scans = self._selected_scans()
        if not scans:
            messagebox.showwarning("No selection", "Select scans for analysis first.")
            return None
        if len(scans) == 1:
            return scans[0]

        dialog = tk.Toplevel(self.root)
        dialog.title("Choose Scan")
        dialog.geometry("900x360")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text="Choose one of the currently selected scans:").pack(anchor="w", padx=10, pady=(10, 4))
        listbox = tk.Listbox(dialog, width=130, height=12, selectmode=tk.SINGLE)
        listbox.pack(fill="both", expand=True, padx=10)
        for idx, scan in enumerate(scans):
            listbox.insert(
                idx,
                self._scan_dialog_label(
                    scan,
                    include_file_timestamp=True,
                    include_group=True,
                ),
            )
        listbox.selection_set(0)
        listbox.focus_set()

        chosen: dict[str, Optional[VNAScan]] = {"scan": None}

        def accept() -> None:
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning("No scan selected", "Choose one scan.", parent=dialog)
                return
            chosen["scan"] = scans[int(selection[0])]
            dialog.destroy()

        btns = tk.Frame(dialog)
        btns.pack(fill="x", padx=10, pady=10)
        tk.Button(btns, text="Cancel", width=12, command=dialog.destroy).pack(side="right")
        tk.Button(btns, text="Open", width=12, command=accept).pack(side="right", padx=(0, 8))
        listbox.bind("<Double-Button-1>", lambda _e: accept())
        dialog.wait_window()
        return chosen["scan"]


    @staticmethod
    def _scan_dialog_path_text(scan: VNAScan) -> str:
        source_dir = str(getattr(scan, "source_dir", "") or "").strip()
        try:
            path = Path(source_dir) if source_dir else Path(str(scan.filename)).resolve().parent
        except Exception:
            path = Path(source_dir) if source_dir else Path(str(scan.filename)).parent
        parts = [part for part in path.parts if part not in (path.anchor, "")]
        if not parts:
            return str(path)
        return str(Path(*parts[-2:])) if len(parts) >= 2 else parts[-1]


    def _scan_dialog_label(
        self,
        scan: VNAScan,
        *,
        index: Optional[int] = None,
        include_file_timestamp: bool = False,
        include_loaded_at: bool = False,
        include_group: bool = False,
    ) -> str:
        prefix = f"{int(index):03d} | " if index is not None else ""
        parts = [prefix + Path(str(scan.filename)).name, f"folder {self._scan_dialog_path_text(scan)}"]
        if include_file_timestamp:
            file_timestamp = str(getattr(scan, "file_timestamp", "")).strip() or "unknown"
            parts.append(f"file {file_timestamp}")
        if include_loaded_at:
            parts.append(f"loaded {scan.loaded_at}")
        if include_group:
            parts.append(f"group {int(scan.plot_group)}" if scan.plot_group is not None else "no group")
        return " | ".join(parts)


    @staticmethod
    def _scan_file_two_level_context(scan: VNAScan) -> str:
        try:
            path = Path(str(scan.filename)).resolve()
        except Exception:
            path = Path(str(scan.filename))
        parts = [part for part in path.parts if part not in (path.anchor, "")]
        if not parts:
            return str(path)
        if len(parts) >= 3:
            return str(Path(*parts[-3:]))
        return str(Path(*parts))


    def load_vna_scan(self) -> None:
        mode = self._choose_vna_load_mode()
        if mode is None:
            return

        path_texts = filedialog.askopenfilenames(
            title="Select VNA scan file(s)",
            filetypes=[
                ("Supported VNA files", "*.npy *.txt *.dat *.csv *.s2p"),
                ("NumPy files", "*.npy"),
                ("Text files", "*.txt *.dat *.csv"),
                ("Touchstone S2P files", "*.s2p"),
                ("All files", "*.*"),
            ],
        )
        if not path_texts:
            return

        added_count = 0
        failed: List[str] = []
        warnings: List[str] = []

        paths = [Path(path_text) for path_text in path_texts]

        def _add_scan(scan: VNAScan) -> None:
            self.dataset.vna_scans.append(scan)
            self.dataset.selected_scan_keys.append(self._scan_key(scan))
            self.dataset.processing_history.append(
                _make_event(
                    "add_vna_scan_to_dataset",
                    {"filename": scan.filename, "loaded_at": scan.loaded_at},
                )
            )

        if mode == "autodetect":
            pair_handled = False
            if len(paths) == 2:
                try:
                    pair_scan, pair_warning = _try_load_vna_npy_pair(paths[0], paths[1])
                except Exception as exc:
                    failed.append(f"{paths[0].name} + {paths[1].name}: {exc}")
                    pair_handled = True
                else:
                    if pair_scan is not None:
                        _add_scan(pair_scan)
                        self._mark_dirty()
                        added_count += 1
                        if pair_warning:
                            warnings.append(pair_warning)
                        pair_handled = True

            if not pair_handled:
                for path in paths:
                    try:
                        scan, warning = _load_vna_file(path)
                        _add_scan(scan)
                        self._mark_dirty()
                        added_count += 1
                        if warning:
                            warnings.append(warning)
                    except Exception as exc:
                        failed.append(f"{path.name}: {exc}")
        else:
            for path in paths:
                try:
                    if path.suffix.lower() != ".npy":
                        raise ValueError("Legacy MHz/dB/deg loader supports only .npy files.")
                    scan = _load_vna_npy_mhz_db_deg(path)
                    warning = None
                    _add_scan(scan)
                    self._mark_dirty()
                    added_count += 1
                    if warning:
                        warnings.append(warning)
                except Exception as exc:
                    failed.append(f"{path.name}: {exc}")

        self._refresh_status()
        self._log(f"VNA load result: added={added_count}, failed={len(failed)}")
        for warning in warnings:
            self._log(f"VNA load warning: {warning}")
        if added_count > 0:
            self._autosave_dataset()

        if added_count > 0 and not failed:
            if warnings:
                messagebox.showwarning(
                    "VNA scans loaded with assumptions",
                    f"Added {added_count} scan(s). Dataset saved automatically.\n\n"
                    + "\n".join(warnings[:8]),
                )
                return
            messagebox.showinfo(
                "VNA scans loaded",
                f"Added {added_count} scan(s). Dataset saved automatically.",
            )
            return

        if added_count > 0 and failed:
            detail_lines = [f"Added {added_count} scan(s), {len(failed)} failed."]
            if warnings:
                detail_lines.extend(["", "Assumption warnings:"])
                detail_lines.extend(warnings[:8])
            detail_lines.extend(["", "Failures:"])
            detail_lines.extend(failed[:8])
            messagebox.showwarning("VNA scan load partial", "\n".join(detail_lines))
            return

        messagebox.showerror(
            "VNA scan load failed",
            "No files were loaded.\n\n" + "\n".join(failed[:8]),
        )


    def _choose_vna_load_mode(self) -> Optional[str]:
        dialog = tk.Toplevel(self.root)
        dialog.title("Load VNA Scan(s)")
        dialog.geometry("880x300")
        dialog.transient(self.root)
        dialog.grab_set()

        mode_var = tk.StringVar(value="autodetect")
        tk.Label(
            dialog,
            text="Choose a load method before selecting file(s):",
            anchor="w",
            justify="left",
        ).pack(anchor="w", padx=12, pady=(12, 8))

        autodetect_text = (
            "Autodetect (default):\n"
            "  - .npy: (3,N)/(N,3) [freq, real, imag]\n"
            "  - .npy: (2,N)/(N,2) [freq, complex]\n"
            "  - two 1D .npy files: [freq] + [complex]\n"
            "  - text: 3-col [freq_Hz, real, imag]\n"
            "  - text: 2-col [freq_MHz, amp_dB] (phase assumed 0)\n"
            "  - .s2p: S21 (RI/MA/DB), freq units HZ/KHZ/MHZ/GHZ"
        )
        legacy_text = (
            "Legacy explicit .npy format:\n"
            "  - (3,N)/(N,3) [freq_MHz, amp_dB, phase_deg]\n"
            "  - use this for files like Be241202p1_hybrid_vna_PE20241213.npy"
        )

        tk.Radiobutton(
            dialog,
            text=autodetect_text,
            value="autodetect",
            variable=mode_var,
            anchor="w",
            justify="left",
            wraplength=840,
        ).pack(anchor="w", padx=12, pady=(0, 8))
        tk.Radiobutton(
            dialog,
            text=legacy_text,
            value="legacy_mhz_db_deg",
            variable=mode_var,
            anchor="w",
            justify="left",
            wraplength=840,
        ).pack(anchor="w", padx=12, pady=(0, 8))

        selection: dict[str, Optional[str]] = {"mode": None}

        def _choose_and_close() -> None:
            picked = str(mode_var.get() or "autodetect").strip().lower()
            if picked not in {"autodetect", "legacy_mhz_db_deg"}:
                picked = "autodetect"
            selection["mode"] = picked
            dialog.destroy()

        button_row = tk.Frame(dialog)
        button_row.pack(fill="x", padx=12, pady=(8, 12))
        tk.Button(button_row, text="Cancel", width=12, command=dialog.destroy).pack(side="right")
        tk.Button(button_row, text="Select File(s)...", width=14, command=_choose_and_close).pack(
            side="right", padx=(0, 8)
        )

        self.root.wait_window(dialog)
        return selection["mode"]


    def remove_vna_scans(self) -> None:
        if not self.dataset.vna_scans:
            messagebox.showwarning("No data", "No VNA scans are loaded in this dataset.")
            return

        selector = tk.Toplevel(self.root)
        selector.title("Remove VNA Scan(s)")
        selector.geometry("820x420")
        selector.transient(self.root)
        selector.grab_set()

        tk.Label(selector, text="Select scan(s) to remove from this dataset:").pack(
            anchor="w", padx=10, pady=(10, 4)
        )
        listbox = tk.Listbox(selector, width=130, height=16, selectmode=tk.MULTIPLE)
        listbox.pack(fill="both", expand=True, padx=10, pady=4)

        for idx, scan in enumerate(self.dataset.vna_scans):
            label = self._scan_dialog_label(scan, index=idx, include_loaded_at=True)
            listbox.insert(tk.END, label)

        def do_remove() -> None:
            indices = sorted(listbox.curselection())
            if not indices:
                selector.destroy()
                return
            names = [Path(self.dataset.vna_scans[i].filename).name for i in indices]
            ok = messagebox.askyesno(
                "Confirm Remove",
                f"Remove {len(indices)} scan(s) from dataset?\n\n"
                + "\n".join(names[:8])
                + ("\n..." if len(names) > 8 else ""),
                parent=selector,
            )
            if not ok:
                return

            removed = []
            for i in reversed(indices):
                scan = self.dataset.vna_scans.pop(i)
                removed.append(scan)

            self.dataset.selected_scan_keys = [
                self._scan_key(scan)
                for scan in self.dataset.vna_scans
                if self._scan_key(scan) in set(self.dataset.selected_scan_keys)
            ]
            self.dataset.processing_history.append(
                _make_event(
                    "remove_vna_scans_from_dataset",
                    {
                        "removed_count": len(removed),
                        "removed_files": [s.filename for s in removed],
                    },
                )
            )
            self._mark_dirty()
            self._refresh_status()
            self._log(f"Removed {len(removed)} VNA scan(s) from dataset.")
            self._autosave_dataset()
            selector.destroy()

        btns = tk.Frame(selector)
        btns.pack(fill="x", padx=10, pady=(4, 10))
        tk.Button(btns, text="Cancel", command=selector.destroy).pack(side="right")
        tk.Button(btns, text="Remove Selected", command=do_remove).pack(side="right", padx=(0, 8))


    def group_selected_scans_for_plotting(self) -> None:
        scans = self._selected_scans()
        if not scans:
            messagebox.showwarning(
                "No selection",
                "No scans selected for analysis.\nUse 'Select Scans for Analysis' first.",
            )
            return

        selector = tk.Toplevel(self.root)
        selector.title("Group Selected Scans")
        selector.geometry("900x460")
        selector.transient(self.root)
        selector.grab_set()

        existing_groups = sorted(
            {int(scan.plot_group) for scan in self.dataset.vna_scans if scan.plot_group is not None}
        )
        prompt = (
            "Choose the subset of currently selected analysis scans to join into one plot group.\n"
            "Each scan can belong to only one group. Enter a positive integer to assign, or leave blank to clear."
        )
        if existing_groups:
            prompt += "\nExisting groups: " + ", ".join(str(g) for g in existing_groups)
        tk.Label(selector, text=prompt, anchor="w", justify="left", wraplength=860).pack(
            anchor="w", padx=10, pady=(10, 6)
        )

        group_frame = tk.Frame(selector)
        group_frame.pack(fill="x", padx=10, pady=(0, 6))
        tk.Label(group_frame, text="Group number:").pack(side="left")
        group_entry = tk.Entry(group_frame, width=12)
        group_entry.pack(side="left", padx=(6, 12))
        tk.Label(group_frame, text="Blank clears group for the chosen subset.").pack(side="left")

        listbox = tk.Listbox(selector, width=160, height=16, selectmode=tk.EXTENDED)
        listbox.pack(fill="both", expand=True, padx=10, pady=4)
        for idx, scan in enumerate(scans):
            label = self._scan_dialog_label(
                scan,
                index=idx,
                include_file_timestamp=True,
                include_loaded_at=True,
                include_group=True,
            )
            listbox.insert(tk.END, label)
            listbox.selection_set(idx)

        def do_apply() -> None:
            indices = sorted(listbox.curselection())
            if not indices:
                messagebox.showwarning("No subset selected", "Select at least one scan in the grouping window.", parent=selector)
                return

            text = group_entry.get().strip()
            new_group: Optional[int] = None
            if text:
                try:
                    new_group = int(text)
                except ValueError:
                    messagebox.showwarning("Invalid group", "Enter a positive integer group number.", parent=selector)
                    return
                if new_group <= 0:
                    messagebox.showwarning("Invalid group", "Enter a positive integer group number.", parent=selector)
                    return

            chosen_scans = [scans[i] for i in indices]
            changed = 0
            for scan in chosen_scans:
                if scan.plot_group != new_group:
                    scan.plot_group = new_group
                    changed += 1

            if changed == 0:
                self._log("Grouping unchanged for selected scan subset.")
                selector.destroy()
                return

            self.dataset.processing_history.append(
                _make_event(
                    "group_selected_scans_for_plotting",
                    {
                        "selected_count": len(chosen_scans),
                        "plot_group": new_group,
                        "filenames": [scan.filename for scan in chosen_scans],
                    },
                )
            )
            self._mark_dirty()
            self._refresh_status()
            self._autosave_dataset()
            if new_group is None:
                self._log(f"Cleared plot groups for {changed} selected scan(s).")
                messagebox.showinfo("Plot groups updated", f"Cleared plot groups for {changed} selected scan(s).", parent=selector)
            else:
                self._log(f"Assigned plot group {new_group} to {changed} selected scan(s).")
                messagebox.showinfo(
                    "Plot groups updated",
                    f"Assigned plot group {new_group} to {changed} selected scan(s).",
                    parent=selector,
                )
            selector.destroy()

        btns = tk.Frame(selector)
        btns.pack(fill="x", padx=10, pady=(4, 10))
        tk.Button(btns, text="Select All", command=lambda: listbox.select_set(0, tk.END)).pack(side="left")
        tk.Button(btns, text="Clear Selection", command=lambda: listbox.selection_clear(0, tk.END)).pack(
            side="left", padx=(6, 0)
        )
        tk.Button(btns, text="Cancel", command=selector.destroy).pack(side="right")
        tk.Button(btns, text="Apply Group", command=do_apply).pack(side="right", padx=(0, 8))


    def clear_selected_scan_attachments(self) -> None:
        scans = self._selected_scans()
        if not scans:
            messagebox.showwarning(
                "No selection",
                "No scans selected for analysis.\nUse 'Select Scans for Analysis' first.",
            )
            return

        selector = tk.Toplevel(self.root)
        selector.title("Clear Attachments")
        selector.geometry("920x480")
        selector.transient(self.root)
        selector.grab_set()

        prompt = (
            "Choose the subset of selected analysis scans to reset.\n"
            "This returns each chosen scan to its original loaded state, while keeping plot group membership."
        )
        tk.Label(selector, text=prompt, anchor="w", justify="left", wraplength=880).pack(
            anchor="w", padx=10, pady=(10, 6)
        )

        listbox = tk.Listbox(selector, width=145, height=18, selectmode=tk.MULTIPLE)
        listbox.pack(fill="both", expand=True, padx=10, pady=4)
        for idx, scan in enumerate(scans):
            label = self._scan_dialog_label(
                scan,
                index=idx,
                include_loaded_at=True,
                include_group=True,
            )
            listbox.insert(tk.END, label)

        def do_clear() -> None:
            indices = sorted(listbox.curselection())
            if not indices:
                messagebox.showwarning(
                    "No subset selected",
                    "Select at least one scan in the clear-attachments window.",
                    parent=selector,
                )
                return

            chosen_scans = [scans[i] for i in indices]
            names = [Path(scan.filename).name for scan in chosen_scans]
            ok = messagebox.askyesno(
                "Confirm Clear",
                f"Clear attachments for {len(chosen_scans)} selected scan(s)?\n\n"
                + "\n".join(names[:8])
                + ("\n..." if len(names) > 8 else ""),
                parent=selector,
            )
            if not ok:
                return

            cleared = 0
            for scan in chosen_scans:
                load_events = [
                    event
                    for event in scan.processing_history
                    if str(event.get("action", "")).startswith("load_vna_")
                ]
                scan.s21_phase_deg_unwrapped = None
                scan.baseline_filter = {}
                scan.candidate_resonators = {}
                scan.processing_history = load_events[:1]
                cleared += 1

            self.dataset.processing_history.append(
                _make_event(
                    "clear_selected_scan_attachments",
                    {
                        "selected_count": len(chosen_scans),
                        "filenames": [scan.filename for scan in chosen_scans],
                    },
                )
            )
            self._mark_dirty()
            self._refresh_status()
            self._autosave_dataset()
            self._log(f"Cleared attachments for {cleared} selected scan(s).")
            messagebox.showinfo(
                "Attachments cleared",
                f"Cleared attachments for {cleared} selected scan(s).\nPlot groups were preserved.",
                parent=selector,
            )
            selector.destroy()

        btns = tk.Frame(selector)
        btns.pack(fill="x", padx=10, pady=(4, 10))
        tk.Button(btns, text="Select All", command=lambda: listbox.select_set(0, tk.END)).pack(side="left")
        tk.Button(btns, text="Clear Selection", command=lambda: listbox.selection_clear(0, tk.END)).pack(
            side="left", padx=(6, 0)
        )
        tk.Button(btns, text="Cancel", command=selector.destroy).pack(side="right")
        tk.Button(btns, text="Clear Attachments", command=do_clear).pack(side="right", padx=(0, 8))
