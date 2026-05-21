import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import ppigrf
import pandas as pd

# =========================
# LOAD DATA
# =========================
data = loadmat('GNSS TEC/inacors_tecproc_14May2024.mat')

lat = data['lat_arr'].flatten()
lon = data['lon_arr'].flatten()
vtec = data['vtec_arr'].flatten()
utc = data['utc_arr'].flatten()

# =========================
# FIX LONGITUDE
# =========================
lon = np.where(lon < 0, lon + 360, lon)

# convert 0–360 to -180–180 for geomag calc
lon_fixed = np.where(lon > 180, lon - 360, lon)

# =========================
# BATAS INDONESIA
# =========================
lat_min, lat_max = -11, 6
lon_min, lon_max = 95, 141

# =========================
# TIME BIN (10 menit)
# =========================
interval = 10 / 60  # jam
time_bin = np.round(utc / interval).astype(int)
time_bin = np.clip(time_bin, 0, 143)

vmin = np.nanmin(vtec)
vmax = np.nanmax(vtec)

# =========================
# GRID GEOMAGNETIK
# =========================
lat_grid = np.linspace(lat_min, lat_max, 40)
lon_grid = np.linspace(lon_min, lon_max, 60)

LON, LAT = np.meshgrid(lon_grid, lat_grid)
MLAT = np.zeros_like(LAT)

date = pd.Timestamp('2024-05-08')

for i in range(LAT.shape[0]):
    for j in range(LAT.shape[1]):

        lon_calc = LON[i, j]
        if lon_calc > 180:
            lon_calc -= 360

        try:
            Be, Bn, Bu = ppigrf.igrf(
                lon_calc, LAT[i, j], 350, date
            )

            Be = Be.item()
            Bn = Bn.item()
            Bu = Bu.item()

            H = np.sqrt(Be**2 + Bn**2)
            I = np.arctan2(-Bu, H)

            MLAT[i, j] = np.degrees(
                np.arctan(0.5 * np.tan(I))
            )

            # print(
            #     "LAT:", LAT[i, j],
            #     "LON:", LON[i, j],
            #     "MLAT:", MLAT[i, j]
            # )

        except:
            MLAT[i, j] = np.nan

print("MLAT min:", np.nanmin(MLAT))
print("MLAT max:", np.nanmax(MLAT))

# =========================
# LOOP PER TIME BIN
# =========================
for t in sorted(np.unique(time_bin)):

    mask_bin = (time_bin == t)
    times_in_bin = np.unique(utc[mask_bin])

    if len(times_in_bin) == 0:
        continue

    t_epoch = times_in_bin[0]
    mask_time = (utc == t_epoch)

    lat_h = lat[mask_time]
    lon_h = lon[mask_time]
    vtec_h = vtec[mask_time]

    if len(lat_h) == 0:
        continue

    mask = (
        (lat_h >= lat_min) & (lat_h <= lat_max) &
        (lon_h >= lon_min) & (lon_h <= lon_max)
    )

    lat_id = lat_h[mask]
    lon_id = lon_h[mask]
    vtec_id = vtec_h[mask]

    if len(lat_id) == 0:
        continue

    # =========================
    # PLOT
    # =========================
    plt.figure(figsize=(20, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, alpha=0.2)
    ax.add_feature(cfeature.OCEAN, alpha=0.2)

    sc = ax.scatter(
        lon_id, lat_id,
        c=vtec_id,
        cmap='jet',
        s=8,
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree()
    )

    cbar = plt.colorbar(sc, pad=0.05, shrink=0.7)
    cbar.set_label(r'$\mathbf{TEC\ [10^{16}/m^2]}$')
    cbar.ax.tick_params(labelsize=8)

    ax.set_extent([lon_min, lon_max, lat_min, lat_max])
    
    ax.set_xlabel("Geographic Longitude")
    ax.set_ylabel("Geographic Latitude")
    
    ax.text(
        0.5, -0.10,
        "Geographic Longitude",
        transform=ax.transAxes,
        ha='center',
        fontsize=11,
        fontweight='bold'
    )

    ax.text(
        -0.08, 0.5,
        "Geographic Latitude",
        transform=ax.transAxes,
        va='center',
        rotation=90,
        fontsize=11,
        fontweight='bold'
    )

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=1,
        color='gray',
        alpha=0.5,
        linestyle='--'
    )

    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10, 'fontweight': 'bold'}
    gl.ylabel_style = {'size': 10, 'fontweight': 'bold'}

    # =========================
    # CONTOUR MLAT
    # =========================
    levels = np.linspace(np.nanmin(MLAT), np.nanmax(MLAT), 6)

    cs = ax.contour(
        LON, LAT, MLAT,
        levels=levels,
        colors='black',
        linewidths=1,
        transform=ccrs.PlateCarree()
    )

    texts = ax.clabel(cs, inline=True, fontsize=10)

    for b in texts:
        b.set_fontweight('bold')

    # =========================
    # TIME LABEL
    # =========================
    hour_label = int((t * 10) // 60) % 24
    minute_label = int((t * 10) % 60)

    plt.title(
        f'TEC Indonesia {hour_label:02d}:{minute_label:02d}',
        fontweight='bold'
    )
    # ========================= # SAVE # ========================= 
    #plt.show()
    plt.savefig(f'BRIN/GNSS Data/TEC_14MAY_10MINS/tec_{hour_label:02d}_{minute_label:02d}.png', dpi=160)
    plt.close()