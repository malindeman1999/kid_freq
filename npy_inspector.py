from __future__ import annotations

import tkinter as tk
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
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

        self._build_ui()
        self._render_empty_output()

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

        output_frame = tk.LabelFrame(right_panel, text="Array Contents", padx=8, pady=8)
        output_frame.pack(fill="both", expand=True)

        self.output_text = tk.Text(output_frame, wrap="none", font=("Consolas", 10))
        self.output_text.grid(row=0, column=0, sticky="nsew")

        y_scroll = ttk.Scrollbar(output_frame, orient="vertical", command=self.output_text.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll = ttk.Scrollbar(output_frame, orient="horizontal", command=self.output_text.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")

        self.output_text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set, state="disabled")
        output_frame.rowconfigure(0, weight=1)
        output_frame.columnconfigure(0, weight=1)

        bottom_frame = tk.Frame(self.root, padx=10, pady=5)
        bottom_frame.pack(side="bottom", fill="x")
        tk.Label(bottom_frame, text="Displays full array contents in a scrollable table view.").pack(anchor="w")

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
        self._render_array_output()

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
        self._render_empty_output()

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
                    command=self._render_array_output,
                )
                spin.pack(side="left")
                spin.bind("<Return>", lambda _e: self._render_array_output())
                spin.bind("<FocusOut>", lambda _e: self._render_array_output())
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

    def _render_empty_output(self) -> None:
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.insert("end", "Load a .npy or .npz file to display array contents")
        self.output_text.configure(state="disabled")

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

    def _format_1d_table(self, arr: np.ndarray) -> str:
        val_strings = [str(v) for v in arr.tolist()]
        idx_width = max(5, len(str(len(arr) - 1)))
        val_width = max(5, max((len(v) for v in val_strings), default=5))
        lines = [f"{'index':>{idx_width}} | {'value':<{val_width}}", f"{'-' * idx_width}-+-{'-' * val_width}"]
        for i, v in enumerate(val_strings):
            lines.append(f"{i:>{idx_width}} | {v:<{val_width}}")
        return "\n".join(lines)

    def _format_2d_table(self, arr: np.ndarray) -> str:
        rows, cols = arr.shape
        str_grid = [[str(arr[r, c]) for c in range(cols)] for r in range(rows)]
        col_widths = []
        for c in range(cols):
            header = f"c{c}"
            content_width = max((len(str_grid[r][c]) for r in range(rows)), default=0)
            col_widths.append(max(len(header), content_width, 6))

        row_idx_width = max(4, len(str(rows - 1)))
        header_cells = [f"{f'c{c}':>{col_widths[c]}}" for c in range(cols)]
        header = f"{'row':>{row_idx_width}} | " + " | ".join(header_cells)
        separator = f"{'-' * row_idx_width}-+-" + "-+-".join("-" * w for w in col_widths)

        lines = [header, separator]
        for r in range(rows):
            cells = [f"{str_grid[r][c]:>{col_widths[c]}}" for c in range(cols)]
            lines.append(f"{r:>{row_idx_width}} | " + " | ".join(cells))
        return "\n".join(lines)

    def _render_array_output(self) -> None:
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", "end")
        if self.data is None:
            self.output_text.insert("end", "No array loaded")
            self.output_text.configure(state="disabled")
            return

        try:
            array = self._get_slice()
            if array is None:
                raise ValueError("No array data available")

            lines = [
                f"dtype: {array.dtype}",
                f"shape: {array.shape}",
                "",
            ]
            if self.data.ndim >= 3:
                indices = []
                for axis in range(self.data.ndim - 2):
                    axis_var = self.slice_controls.get(axis)
                    indices.append(str(min(max(0, axis_var.get()), self.data.shape[axis] - 1)))
                lines.extend([f"source slice indices (leading axes): ({', '.join(indices)})", ""])

            if array.ndim == 1:
                lines.append(self._format_1d_table(array))
            elif array.ndim == 2:
                lines.append(self._format_2d_table(array))
            else:
                lines.append(np.array2string(array, threshold=200, edgeitems=3))

            self.output_text.insert("end", "\n".join(lines))
        except Exception as exc:
            self.output_text.insert("end", f"Unable to render array contents:\n{exc}")
        self.output_text.configure(state="disabled")


def main() -> None:
    root = tk.Tk()
    app = NpyInspector(root)
    root.mainloop()


if __name__ == "__main__":
    main()
