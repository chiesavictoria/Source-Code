#INI UNTUK CODINGAN YANG NGEBUANG DATA TIDAK VALID

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob

#ambil semua data file csv
files = sorted(glob.glob("Plot Data Infrasound/240511/*.csv"))

# print(files)

df_list = []
for file in files:
    df = pd.read_csv(file, header=None)
    df_list.append(df)

data_raw = pd.concat(df_list, ignore_index=True)

#FILTERING DATA AMBIL YANG VALID
data = data_raw[
    (data_raw[14] >= 1900) & (data_raw[14] <= 2100) &
    (data_raw[15] >= 1) & (data_raw[15] <= 12) &
    (data_raw[16] >= 1) & (data_raw[16] <= 31)
].reset_index(drop=True)

print(data)

time = pd.to_datetime({
    "year": data[14],
    "month": data[15],
    "day": data[16]
})

print(time)

timeTrue = (
    pd.to_datetime(time, format="%Y%m%d")
    + pd.to_timedelta(data[17], unit='h')
    + pd.to_timedelta(data[18], unit='m')
    + pd.to_timedelta(data[19], unit='s')
)
tanggal_label = f"{int(data[14].iloc[0])}{int(data[15].iloc[0]):02d}{int(data[16].iloc[0]):02d}"

plt.figure(figsize=(14,10))
plt.plot(timeTrue, data[5], label=tanggal_label)

plt.xlabel("Time [UTC]", fontweight='bold')
plt.ylabel("IS pressure [mPa]", fontweight='bold')

plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.margins(x=0)

plt.legend(ncol=3, fontsize=8, prop={'weight':'bold'})
# Format x-axis jadi HH:MM
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.grid()

plt.show()