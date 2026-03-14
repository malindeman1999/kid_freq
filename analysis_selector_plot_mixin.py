from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import messagebox

from analysis_io import _dataset_dir
from analysis_models import _make_event


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
            label = f"{idx:03d} | {Path(scan.filename).name} | loaded {scan.loaded_at}"
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

        n = len(scans)
        fig, axes = plt.subplots(n, 2, figsize=(14, max(4, 3 * n)), squeeze=False)

        for row, scan in enumerate(scans):
            amp = scan.amplitude()
            if scan.has_dewrapped_phase():
                phase = scan.phase_deg_unwrapped()
                phase_label = "Phase (dewrapped, deg)"
            else:
                phase = scan.phase_deg_wrapped_raw()
                phase_label = "Phase (raw wrapped, deg)"
            phase_min = np.min(phase)
            phase_max = np.max(phase)
            phase_span = phase_max - phase_min
            phase_pad = 1.0 if phase_span == 0 else 0.05 * phase_span

            ax_amp = axes[row][0]
            ax_phase = axes[row][1]
            ax_amp.plot(scan.freq, amp, color="tab:green")
            ax_amp.set_ylabel("|S21|")
            ax_amp.grid(True, alpha=0.3)
            ax_amp.set_title(Path(scan.filename).name, fontsize=10)

            ax_phase.plot(scan.freq, phase, color="tab:red")
            ax_phase.set_ylabel(phase_label)
            ax_phase.set_ylim(phase_min - phase_pad, phase_max + phase_pad)
            ax_phase.grid(True, alpha=0.3)

            if row == n - 1:
                ax_amp.set_xlabel("Frequency")
                ax_phase.set_xlabel("Frequency")

        fig.suptitle("Selected VNA Scans", fontsize=12)
        plt.tight_layout()

        pdf_dir = _dataset_dir(self.dataset)
        pdf_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = pdf_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_selected_vna_panels.pdf"
        fig.savefig(pdf_path)
        self.dataset.processing_history.append(
            _make_event(
                "plot_selected_vna_scans",
                {"selected_count": n, "pdf_path": str(pdf_path)},
            )
        )
        self._mark_dirty()
        self._log(f"Plotted {n} selected scan(s). PDF saved: {pdf_path}")
        self._autosave_dataset()

        plt.show()
