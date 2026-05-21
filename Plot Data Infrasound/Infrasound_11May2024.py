import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
from scipy.signal import spectrogram, butter, filtfilt

# =========================
# 1. Load & gabung data
# =========================
files = sorted(glob.glob("Plot Data Infrasound/240511/*.csv"))
df_list = [pd.read_csv(f, header=None) for f in files]
data_raw = pd.concat(df_list, ignore_index=True)

# =========================
# 2. Filtering data valid
# =========================
data = data_raw[
    (data_raw[14] >= 1900) & (data_raw[14] <= 2100) &
    (data_raw[15] >= 1) & (data_raw[15] <= 12) &
    (data_raw[16] >= 1) & (data_raw[16] <= 31)
].reset_index(drop=True)

# =========================
# 3. Parsing waktu lengkap
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
# 4. Signal
# =========================
signal = data[5].astype(float).values
Fs = 4

# =========================
# 5. Bandpass Filter Function
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
# 6. Spectrogram
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
# 7. Convert dB
# =========================
Sxx_dB = 10 * np.log10(Sxx + 1e-12)

# =========================
# 8. Mapping waktu spectrogram
# =========================
indices = np.round(t * Fs).astype(int)
indices = np.clip(indices, 0, len(timeTrue) - 1)
time_spec = timeTrue.iloc[indices]

# =========================
# 9. Dynamic range
# =========================
vmin = np.percentile(Sxx_dB, 10)
vmax = np.percentile(Sxx_dB, 90)

# =========================
# 10. Combined Plot
# =========================
fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(15,10),
    sharex=True,
    gridspec_kw={'height_ratios': [1, 1.2]}
)

# -------------------------
# Time Series Plot
# -------------------------
ax1.plot(
    timeTrue,
    signal,
    color='royalblue',
    linewidth=0.8,
    label=tanggal_label
)

ax1.set_ylabel("IS pressure [mPa]", fontweight='bold')
ax1.legend(loc='upper right', prop={'weight':'bold'})
ax1.grid(True)

# -------------------------
# Spectrogram Plot
# -------------------------
mesh = ax2.pcolormesh(
    time_spec,
    f,
    Sxx_dB,
    shading='auto',
    cmap='inferno',
    vmin=vmin,
    vmax=vmax
)

ax2.set_ylabel("Frequency [Hz]", fontweight='bold')
ax2.set_xlabel("Time [UTC]", fontweight='bold')
ax2.set_ylim(0, 2)
ax2.grid(False)

# Sinkronisasi sumbu
ax1.margins(x=0)
ax2.margins(x=0)

ax1.set_xlim(timeTrue.iloc[0], timeTrue.iloc[-1])
ax2.set_xlim(timeTrue.iloc[0], timeTrue.iloc[-1])

# Colorbar
cbar = fig.colorbar(mesh, ax=ax2)
cbar.set_label("Power (dB)", fontweight='bold')

# =========================
# 11. Format Axis
# =========================
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

ax1.tick_params(axis='x', labelbottom=True)

for ax in [ax1, ax2]:
    ax.tick_params(axis='both', labelsize=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

plt.subplots_adjust(hspace=0.05)
plt.tight_layout()
plt.show()