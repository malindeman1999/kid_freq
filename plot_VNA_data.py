import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tkinter import Tk, filedialog

initial_dir = Path(
    r"C:\Users\lindeman\OneDrive - JPL\Documents\PRIMA-related\PRIMA SUBPROJECTS\RELATIVE CHANGES IN RESONANCES\VNA data\from Chris\freq_shift_for_Logan\20240821"
)

root = Tk()
root.withdraw()
selected_file = filedialog.askopenfilename(
    title="Select VNA data file",
    initialdir=str(initial_dir),
    filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")],
)
root.destroy()

if not selected_file:
    raise SystemExit("No file selected.")

data = np.load(selected_file)
print(f"Loaded: {selected_file}")

print(type(data))
print(data.shape)
print(data.dtype)

if data.ndim != 2 or data.shape[0] < 3:
    raise ValueError(f"Expected new format shape (3, N) with rows [freq, real, imag], got {data.shape}")

freq = data[0, :]
s21_real = data[1, :]
s21_imag = data[2, :]
s21_complex = s21_real + 1j * s21_imag
s21_amp = np.abs(s21_complex)
s21_phase_deg = np.degrees(np.unwrap(np.angle(s21_complex)))
phase_min = np.min(s21_phase_deg)
phase_max = np.max(s21_phase_deg)
phase_span = phase_max - phase_min
phase_pad = 1.0 if phase_span == 0 else 0.05 * phase_span

print("Using layout: (3, N) rows")
print("Head [freq, real, imag]:")
print(np.column_stack((freq[:5], s21_real[:5], s21_imag[:5])))

fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 10))
axes[0].plot(freq, s21_real, label="Real(S21)", color="tab:blue")
axes[0].set_ylabel("Real(S21)")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(freq, s21_imag, label="Imag(S21)", color="tab:orange")
axes[1].set_ylabel("Imag(S21)")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

axes[2].plot(freq, s21_amp, label="|S21|", color="tab:green")
axes[2].set_ylabel("|S21|")
axes[2].grid(True, alpha=0.3)
axes[2].legend()

axes[3].plot(freq, s21_phase_deg, label="Phase(S21)", color="tab:red")
axes[3].set_xlabel("Frequency")
axes[3].set_ylabel("Phase (deg)")
axes[3].set_ylim(phase_min - phase_pad, phase_max + phase_pad)
axes[3].grid(True, alpha=0.3)
axes[3].legend()

plt.tight_layout()
plt.show()
