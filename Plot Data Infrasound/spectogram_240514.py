import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.signal import butter, filtfilt

# =========================
# 1. Load data
# =========================
files = sorted(glob.glob("Plot Data Infrasound/240514/*.csv"))
df_list = [pd.read_csv(f, header=None) for f in files]
data = pd.concat(df_list, ignore_index=True)

# =========================
# 2. Parsing waktu
# =========================
time_str = data[0].astype(str) + ":" + data[1].astype(int).astype(str).str.zfill(2)
time = pd.to_datetime("2024-05-10 " + time_str, format="%Y-%m-%d %H:%M:%S")

# =========================
# 3. Signal
# =========================
signal = data[5].astype(float).values
Fs = 4

# =========================
# 4. Bandpass filter
# =========================
def bandpass(x, fs, low, high, order=4):
    nyq = 0.5 * fs
    lowcut = low / nyq
    highcut = high / nyq
    b, a = butter(order, [lowcut, highcut], btype='band')
    return filtfilt(b, a, x)

#memakai filter untuk analisis
signal_filt = signal

# =========================
# 5. Parameter FFT
# =========================
nfft = 256
noverlap = 128

# =========================
# 6. Plot spectrogram (MATPLOTLIB)
# =========================
plt.figure(figsize=(14,6))

Pxx, freqs, bins, im = plt.specgram(
    signal_filt,
    NFFT=nfft,
    Fs=Fs,
    noverlap=noverlap,
    scale='dB'
)

plt.clf()

# mapping ke UTC
indices = (bins * Fs).astype(int)
indices = np.clip(indices, 0, len(time)-1)
time_spec = time.iloc[indices]

plt.figure(figsize=(14,6))
plt.pcolormesh(time_spec, freqs, 10*np.log10(Pxx+1e-12),
               shading='auto', cmap='inferno')

plt.xlabel("Time [UTC]")
plt.ylabel("Frequency [Hz]")
plt.title(f"Spectrogram (NFFT={nfft}, Overlap={noverlap})")

plt.ylim(0, 2)
plt.colorbar(label="Power (dB)")

plt.tight_layout()
plt.show()