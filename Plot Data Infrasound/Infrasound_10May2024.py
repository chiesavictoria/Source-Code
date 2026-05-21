import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
from scipy.signal import butter, filtfilt

# =========================
# 1. Load data
# =========================
files = sorted(glob.glob("Plot Data Infrasound/240510/*.csv"))
df_list = [pd.read_csv(f, header=None) for f in files]
data = pd.concat(df_list, ignore_index=True)

# =========================
# 2. Parsing waktu (Time Series)
# =========================
time = pd.to_datetime({
    "year": data[14],
    "month": data[15],
    "day": data[16]
})

timeTrue = (
    pd.to_datetime(time, format="%Y%m%d")
    + pd.to_timedelta(data[17], unit='h')
    + pd.to_timedelta(data[18], unit='m')
    + pd.to_timedelta(data[19], unit='s')
)

tanggal_label = f"{int(data[14].iloc[0])}{int(data[15].iloc[0]):02d}{int(data[16].iloc[0]):02d}"

# =========================
# 3. Signal
# =========================
signal = data[5].astype(float).values
Fs = 4

# =========================
# 4. Bandpass Filter Function
# =========================
def bandpass(x, fs, low, high, order=4):
    nyq = 0.5 * fs
    lowcut = low / nyq
    highcut = high / nyq
    b, a = butter(order, [lowcut, highcut], btype='band')
    return filtfilt(b, a, x)

# Optional filter
signal_filt = signal

# =========================
# 5. Spectrogram Parameter
# =========================
nfft = 256
noverlap = 128

# =========================
# 6. Generate Spectrogram Data
# =========================
plt.figure()
Pxx, freqs, bins, im = plt.specgram(
    signal_filt,
    NFFT=nfft,
    Fs=Fs,
    noverlap=noverlap,
    scale='dB'
)
plt.close()

# Mapping bins ke UTC
indices = (bins * Fs).astype(int)
indices = np.clip(indices, 0, len(timeTrue)-1)
time_spec = timeTrue.iloc[indices]

# =========================
# 7. Combined Plot
# =========================
fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(14,10),
    sharex=True,
    gridspec_kw={'height_ratios': [1, 1.2]}
)

# -------------------------
# Time Series Plot
# -------------------------
ax1.plot(timeTrue, signal, linewidth=0.8, label=tanggal_label)

ax1.set_ylabel("IS pressure [mPa]", fontweight='bold')
ax1.legend(loc='upper right', prop={'weight':'bold'})
ax1.grid(True)
ax1.margins(x=0)

# -------------------------
# Spectrogram Plot
# -------------------------
pcm = ax2.pcolormesh(
    time_spec,
    freqs,
    10*np.log10(Pxx + 1e-12),
    shading='auto',
    cmap='inferno'
)

ax2.set_ylabel("Frequency [Hz]", fontweight='bold')
ax2.set_xlabel("Time [UTC]", fontweight='bold')
ax2.set_ylim(0, 2)
ax2.grid(False)

#Supaya tidak ada offset kanan kiri
ax1.margins(x=0)
ax2.margins(x=0)

ax1.set_xlim(timeTrue.iloc[0], timeTrue.iloc[-1])
ax2.set_xlim(timeTrue.iloc[0], timeTrue.iloc[-1])

# Colorbar
cbar = fig.colorbar(pcm, ax=ax2)
cbar.set_label("Power (dB)", fontweight='bold')

# =========================
# 8. Format Axis
# =========================
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

ax1.tick_params(axis='x', labelbottom=True)

for ax in [ax1, ax2]:
    ax.tick_params(axis='both', labelsize=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

plt.tight_layout()
plt.show()