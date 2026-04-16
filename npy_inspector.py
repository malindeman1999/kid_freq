from __future__ import annotations

import tkinter as tk
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from tkinter import filedialog, messagebox, ttk


class NpyInspector:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("NumPy Array Inspector")
        self.root.geometry("1100x700")

        self.file_path: Optional[Path] = None
        self.data: Optional[np.ndarray] = None
        self.npz_data: Optional[np.lib.npyio.NpzFile] = None
        self.array_names: List[str] = []

        self.file_var = tk.StringVar(value="No file loaded")
        self.shape_var = tk.StringVar(value="Shape: -")
        self.dtype_var = tk.StringVar(value="Dtype: -")
        self.size_var = tk.StringVar(value="Size: -")
        self.stats_var = tk.StringVar(value="Statistics: -")
        self.preview_text = tk.StringVar(value="Preview unavailable")

        self.slice_controls: Dict[int, tk.IntVar] = {}
        self.slice_selectors: List[ttk.Spinbox] = []

        self.figure: Figure = Figure(figsize=(6.5, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas: Optional[FigureCanvasTkAgg] = None

        self._build_ui()
        self._draw_empty_plot()

    def _build_ui(self) -> None:
        toolbar_frame = tk.Frame(self.root, padx=10, pady=10)
        toolbar_frame.pack(side="top", fill="x")

        tk.Button(toolbar_frame, text="Open .npy / .npz", command=self.open_file, width=18).pack(side="left")
        tk.Button(toolbar_frame, text="Reload", command=self.reload_file, width=12).pack(side="left", padx=(8, 0))

        tk.Label(toolbar_frame, textvariable=self.file_var, anchor="w", justify="left").pack(side="left", padx=(20, 0), fill="x", expand=True)

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill="both", expand=True)

        left_panel = tk.Frame(main_frame, padx=10, pady=10)
        left_panel.pack(side="left", fill="y")

        right_panel = tk.Frame(main_frame, padx=10, pady=10)
        right_panel.pack(side="left", fill="both", expand=True)

        info_frame = tk.LabelFrame(left_panel, text="Array Info", padx=10, pady=10)
        info_frame.pack(fill="x", expand=False)

        for label_text, var in [
            ("Shape:", self.shape_var),
            ("Type:", self.dtype_var),
            ("Size:", self.size_var),
            ("Stats:", self.stats_var),
        ]:
            frame = tk.Frame(info_frame)
            frame.pack(fill="x", pady=2)
            tk.Label(frame, text=label_text, width=8, anchor="w").pack(side="left")
            tk.Label(frame, textvariable=var, anchor="w", justify="left", wraplength=260).pack(side="left", fill="x", expand=True)

        self.slice_frame = tk.LabelFrame(left_panel, text="Slice Selection", padx=10, pady=10)
        self.slice_frame.pack(fill="x", expand=False, pady=(10, 0))

        preview_frame = tk.LabelFrame(left_panel, text="Preview", padx=10, pady=10)
        preview_frame.pack(fill="both", expand=True, pady=(10, 0))
        self.preview_box = tk.Text(preview_frame, width=42, height=22, wrap="none")
        self.preview_box.pack(fill="both", expand=True)
        self.preview_box.configure(state="disabled")

        plot_frame = tk.Frame(right_panel)
        plot_frame.pack(fill="both", expand=True)

        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()

        bottom_frame = tk.Frame(self.root, padx=10, pady=5)
        bottom_frame.pack(side="bottom", fill="x")
        tk.Label(bottom_frame, text="Supported: 1D line plots, 2D heatmaps, first slice of 3D+ arrays.").pack(anchor="w")

    def open_file(self) -> None:
        filename = filedialog.askopenfilename(
            title="Open NumPy file",
            filetypes=[("NumPy files", "*.npy *.npz"), ("All files", "*")],
        )
        if not filename:
            return
        self.file_path = Path(filename)
        self._load_file()

    def reload_file(self) -> None:
        if self.file_path is None:
            messagebox.showinfo("Reload", "No file loaded to reload.")
            return
        self._load_file()

    def _load_file(self) -> None:
        if self.file_path is None:
            return

        self.file_var.set(str(self.file_path))
        self.data = None
        self.npz_data = None
        self.array_names = []
        self.slice_controls.clear()
        self.slice_selectors.clear()

        try:
            if self.file_path.suffix.lower() == ".npz":
                self.npz_data = np.load(str(self.file_path), allow_pickle=True)
                self.array_names = list(self.npz_data.files)
                if not self.array_names:
                    raise ValueError("Empty .npz archive")
                self._build_npz_selector()
                self._set_array(self.array_names[0])
            else:
                self._clear_npz_selector()
                self.data = np.load(str(self.file_path), allow_pickle=True)
                self._refresh_ui()
        except Exception as exc:
            messagebox.showerror("Load failed", f"Unable to load file:\n{exc}")
            self._clear_ui()

    def _build_npz_selector(self) -> None:
        for widget in self.slice_frame.winfo_children():
            widget.destroy()

        tk.Label(self.slice_frame, text="Select array:").pack(anchor="w")
        self.array_selector = ttk.Combobox(self.slice_frame, values=self.array_names, state="readonly")
        self.array_selector.set(self.array_names[0])
        self.array_selector.pack(fill="x", pady=(0, 4))
        self.array_selector.bind("<<ComboboxSelected>>", self._on_npz_selection)

        self._build_slice_controls()

    def _clear_npz_selector(self) -> None:
        for widget in self.slice_frame.winfo_children():
            widget.destroy()

    def _on_npz_selection(self, event: Any) -> None:
        if isinstance(event, tk.Event):
            self._set_array(self.array_selector.get())

    def _set_array(self, array_name: str) -> None:
        if self.npz_data is None:
            return
        try:
            self.data = self.npz_data[array_name]
            self._refresh_ui()
        except Exception as exc:
            messagebox.showerror("Invalid array", f"Unable to load array '{array_name}':\n{exc}")

    def _refresh_ui(self) -> None:
        self._build_slice_controls()
        self._update_info()
        self._update_preview()
        self._draw_plot()

    def _clear_ui(self) -> None:
        self.file_var.set("No file loaded")
        self.shape_var.set("Shape: -")
        self.dtype_var.set("Dtype: -")
        self.size_var.set("Size: -")
        self.stats_var.set("Statistics: -")
        self.preview_box.configure(state="normal")
        self.preview_box.delete("1.0", "end")
        self.preview_box.insert("end", "Preview unavailable")
        self.preview_box.configure(state="disabled")
        self._draw_empty_plot()

    def _build_slice_controls(self) -> None:
        for widget in self.slice_frame.winfo_children():
            widget.destroy()

        if self.npz_data is not None and self.array_names:
            tk.Label(self.slice_frame, text="Select array:").pack(anchor="w")
            self.array_selector = ttk.Combobox(self.slice_frame, values=self.array_names, state="readonly")
            self.array_selector.set(self.array_names[0])
            self.array_selector.pack(fill="x", pady=(0, 4))
            self.array_selector.bind("<<ComboboxSelected>>", self._on_npz_selection)

        if self.data is None:
            return

        shape = self.data.shape
        if self.data.ndim <= 2:
            if self.npz_data is None:
                tk.Label(self.slice_frame, text="No slice controls for 1D/2D arrays.").pack(anchor="w")
            return

        if self.data.ndim >= 3:
            tk.Label(self.slice_frame, text="Slice along first axes:").pack(anchor="w")
            for axis in range(self.data.ndim - 2):
                axis_var = self.slice_controls.get(axis)
                if axis_var is None:
                    axis_var = tk.IntVar(value=0)
                    self.slice_controls[axis] = axis_var
                frame = tk.Frame(self.slice_frame)
                frame.pack(fill="x", pady=2)
                tk.Label(frame, text=f"Axis {axis} (0..{shape[axis]-1}):", width=16, anchor="w").pack(side="left")
                spin = ttk.Spinbox(
                    frame,
                    from_=0,
                    to=max(0, shape[axis] - 1),
                    textvariable=axis_var,
                    width=8,
                    command=self._draw_plot,
                )
                spin.pack(side="left")
                spin.bind("<Return>", lambda _e: self._draw_plot())
                spin.bind("<FocusOut>", lambda _e: self._draw_plot())
                self.slice_selectors.append(spin)

    def _update_info(self) -> None:
        if self.data is None:
            self.shape_var.set("Shape: -")
            self.dtype_var.set("Dtype: -")
            self.size_var.set("Size: -")
            self.stats_var.set("Statistics: -")
            return

        shape_text = f"{self.data.shape}"
        dtype_text = f"{self.data.dtype}"
        size_text = f"{self.data.size} elements, {self.data.nbytes} bytes"
        self.shape_var.set(shape_text)
        self.dtype_var.set(dtype_text)
        self.size_var.set(size_text)

        if np.issubdtype(self.data.dtype, np.number):
            try:
                stats_text = (
                    f"min={float(np.nanmin(self.data)):.6g}, "
                    f"max={float(np.nanmax(self.data)):.6g}, "
                    f"mean={float(np.nanmean(self.data)):.6g}, "
                    f"std={float(np.nanstd(self.data)):.6g}"
                )
            except Exception:
                stats_text = "Unable to compute statistics"
        else:
            stats_text = "Non-numeric array"
        self.stats_var.set(stats_text)

    def _update_preview(self) -> None:
        self.preview_box.configure(state="normal")
        self.preview_box.delete("1.0", "end")
        if self.data is None:
            self.preview_box.insert("end", "No array loaded")
            self.preview_box.configure(state="disabled")
            return

        try:
            if self.data.ndim <= 2 and self.data.size <= 10000:
                self.preview_box.insert("end", np.array2string(self.data, threshold=200, edgeitems=3))
            else:
                summary = [
                    f"dtype: {self.data.dtype}",
                    f"shape: {self.data.shape}",
                    f"size: {self.data.size}",
                ]
                if self.data.ndim >= 3:
                    summary.append("Preview: first 2D slice")
                self.preview_box.insert("end", "\n".join(summary))
        except Exception as exc:
            self.preview_box.insert("end", f"Unable to preview array: {exc}")
        self.preview_box.configure(state="disabled")

    def _draw_empty_plot(self) -> None:
        self.ax.clear()
        self.ax.text(0.5, 0.5, "Load a .npy or .npz file to display array contents", ha="center", va="center")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw_idle()

    def _get_slice(self) -> Optional[np.ndarray]:
        if self.data is None:
            return None
        if self.data.ndim <= 2:
            return self.data

        selection = [0] * self.data.ndim
        for axis in range(self.data.ndim - 2):
            axis_var = self.slice_controls.get(axis)
            selection[axis] = min(max(0, axis_var.get()), self.data.shape[axis] - 1)
        selection[-2] = slice(None)
        selection[-1] = slice(None)
        return self.data[tuple(selection)]

    def _draw_plot(self) -> None:
        self.ax.clear()
        if self.data is None:
            self._draw_empty_plot()
            return

        try:
            array = self._get_slice()
            if array is None:
                raise ValueError("No array to plot")

            if array.ndim != 2 and array.ndim != 1:
                raise ValueError(f"Unsupported plot shape: {array.shape}")

            if array.ndim == 1:
                self.ax.plot(array, marker="o", linestyle="-", markersize=4)
                self.ax.set_title("1D array")
                self.ax.set_xlabel("Index")
                self.ax.set_ylabel("Value")
            else:
                im = self.ax.imshow(array, aspect="auto", interpolation="nearest", cmap="viridis")
                self.figure.colorbar(im, ax=self.ax, fraction=0.046, pad=0.04)
                self.ax.set_title("2D array slice")
                self.ax.set_xlabel("Column")
                self.ax.set_ylabel("Row")

            self.figure.tight_layout()
            self.canvas.draw_idle()
        except Exception as exc:
            self.ax.clear()
            self.ax.text(0.5, 0.5, f"Unable to render plot:\n{exc}", ha="center", va="center", wrap=True)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.canvas.draw_idle()


def main() -> None:
    root = tk.Tk()
    app = NpyInspector(root)
    root.mainloop()


if __name__ == "__main__":
    main()
