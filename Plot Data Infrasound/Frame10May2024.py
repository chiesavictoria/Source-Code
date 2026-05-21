import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.signal import butter, filtfilt

# =========================
# 1. Load data
# =========================
files = sorted(glob.glob("Plot Data Infrasound/240510/*.csv"))
df_list = [pd.read_csv(f, header=None) for f in files]
data = pd.concat(df_list, ignore_index=True)

# =========================
# 2. Parsing waktu
# =========================
time_str = data[0].astype(str) + ":" + data[1].astype(int).astype(str).str.zfill(2)
time = pd.to_datetime("2024-05-10 " + time_str, format="%Y-%m-%d %H:%M:%S")

# =========================
# 3. Ambil sinyal
# =========================
signal = data[5].astype(float).values
Fs = 4  # Sampling frequency (Hz)

# =========================
# 4. Bandpass filter
# =========================
def bandpass(x, fs, low, high, order=4):
    nyq = 0.5 * fs
    lowcut = low / nyq
    highcut = high / nyq
    b, a = butter(order, [lowcut, highcut], btype='band')
    return filtfilt(b, a, x)

# Gunakan filter sesuai kebutuhan
signal_filt = bandpass(signal, Fs, 0.01, 1.5)

# =========================
# 5. Spectrogram parameters
# =========================
nfft = 256
noverlap = 128

# =========================
# 6. Generate spectrogram data
# =========================
Pxx, freqs, bins, _ = plt.specgram(
    signal_filt,
    NFFT=nfft,
    Fs=Fs,
    noverlap=noverlap,
    scale='dB'
)
plt.close()

# =========================
# 7. Mapping waktu spectrogram
# =========================
indices = (bins * Fs).astype(int)
indices = np.clip(indices, 0, len(time)-1)
time_spec = time.iloc[indices]

# =========================
# 8. Normalisasi dB
# =========================
Sxx_dB = 10 * np.log10(Pxx + 1e-12)
Sxx_dB = Sxx_dB - np.max(Sxx_dB)

vmin = np.percentile(Sxx_dB, 5)
vmax = np.percentile(Sxx_dB, 98)

# =========================
# 9. Plot gabungan 1 frame
# =========================
fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(16, 10),
    sharex=True,
    gridspec_kw={'height_ratios': [1, 2]}
)

# -------------------------
# Time Series
# -------------------------
ax1.plot(time, signal_filt, color='black', linewidth=0.7)
ax1.set_ylabel("Amplitude", fontweight='bold')
ax1.set_title("Infrasound Time Series and Spectrogram - 10 May 2024", fontweight='bold')
ax1.grid(True, alpha=0.3)

# -------------------------
# Spectrogram
# -------------------------
pcm = ax2.pcolormesh(
    time_spec,
    freqs,
    Sxx_dB,
    shading='auto',
    cmap='inferno',
    vmin=vmin,
    vmax=vmax
)

ax2.set_ylabel("Frequency [Hz]", fontweight='bold')
ax2.set_xlabel("Time [UTC]", fontweight='bold')
ax2.set_ylim(0, 2)

# Colorbar
cbar = fig.colorbar(pcm, ax=ax2)
cbar.set_label("Power (dB)", fontweight='bold')

# =========================
# 10. Layout
# =========================
plt.tight_layout()
plt.show()