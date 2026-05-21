import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob

#ambil semua data file csv
files = sorted(glob.glob("Plot Data Infrasound/240516/*.csv"))

# print(files)

df_list = []
for file in files:
    df = pd.read_csv(file, header=None)
    df_list.append(df)

data = pd.concat(df_list, ignore_index=True)

print(data)

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