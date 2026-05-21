import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.signal import spectrogram, butter, filtfilt

# =========================
# 1. Load & gabung data
# =========================
files = sorted(glob.glob("Plot Data Infrasound/240511/*.csv"))
df_list = [pd.read_csv(f, header=None) for f in files]
data_raw = pd.concat(df_list, ignore_index=True)

data = data_raw[
    (data_raw[14] >= 1900) & (data_raw[14] <= 2100) &
    (data_raw[15] >= 1) & (data_raw[15] <= 12) &
    (data_raw[16] >= 1) & (data_raw[16] <= 31)
].reset_index(drop=True)
print(data)

# =========================
# 2. Parsing waktu (kolom 0 = jam:menit, kolom 1 = detik)
# =========================
time_str = data[0].astype(str) + ":" + data[1].astype(int).astype(str).str.zfill(2)
time = pd.to_datetime("2024-05-11 " + time_str, format="%Y-%m-%d %H:%M:%S")

# =========================
# 3. Ambil sinyal infrasound
# =========================
signal = data[5].astype(float).values
Fs = 4  # Hz (sampling rate kamu)

# signal = signal[::2]
# time = time[::2]
# Fs = Fs / 2

# =========================
# 4. Bandpass filter (INFRASOUND)
# =========================
def bandpass(x, fs, low, high, order=4):
    nyq = 0.5 * fs
    lowcut = low / nyq
    highcut = high / nyq
    b, a = butter(order, [lowcut, highcut], btype='band')
    return filtfilt(b, a, x)

# filter 0.01–2 Hz (bisa diubah)
signal_filt = signal

# =========================
# 5. Spectrogram (SCIPY)
# =========================

f, t, Sxx = spectrogram(
    signal_filt,
    fs=Fs,
    window='hann',
    nperseg=256,
    noverlap=128,
    detrend='constant',
    scaling='density',
    mode='psd'
)

# =========================
# 6. Konversi ke dB
# =========================
Sxx_dB = 10 * np.log10(Sxx + 1e-12)

# =========================
# 7. Mapping waktu ke UTC
# =========================
indices = np.round(t * Fs).astype(int)
indices = np.clip(indices, 0, len(time) - 1)
time_spec = time.iloc[indices]

# =========================
# 8. Dynamic range biar kontras
# =========================
vmin = np.percentile(Sxx_dB, 10)
vmax = np.percentile(Sxx_dB, 90)

# =========================
# 9. Plot FINAL (clean & profesional)
# =========================
plt.figure(figsize=(15,6))

mesh = plt.pcolormesh(
    time_spec,
    f,
    Sxx_dB,
    shading='auto',
    cmap='inferno',   # paling bagus buat infrasound
    vmin=vmin,
    vmax=vmax
)

plt.xlabel("Time [UTC]", fontsize=12, fontweight='bold')
plt.ylabel("Frequency [Hz]", fontsize=12, fontweight='bold')
plt.title("Infrasound Spectrogram - 11 May 2024", fontsize=14, fontweight='bold')

plt.ylim(0, 2)  # fokus infrasound
# plt.grid(alpha=0.3)

cbar = plt.colorbar(mesh)
cbar.set_label("Power (dB)")

plt.tight_layout()
plt.show()