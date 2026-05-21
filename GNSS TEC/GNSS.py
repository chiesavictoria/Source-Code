import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# =========================
# LOAD DATA
# =========================
data = loadmat('GNSS TEC/inacors_tecproc_08May2024.mat')

lat = data['lat_arr'].flatten()
lon = data['lon_arr'].flatten()
vtec = data['vtec_arr'].flatten()
utc = data['utc_arr'].flatten()

# =========================
# FIX LONGITUDE
# =========================
lon = np.where(lon < 0, lon + 360, lon)

# =========================
# BATAS INDONESIA
# =========================
lat_min, lat_max = -11, 6
lon_min, lon_max = 95, 141

# =========================
# AMBIL EPOCH UNIK
# =========================
unique_time = np.unique(utc)

# =========================
# PILIH 1 EPOCH TERDEKAT TIAP JAM
# =========================
selected_time = []
for h in range(24):
    idx = np.argmin(np.abs(unique_time - h))
    selected_time.append(unique_time[idx])
vmin = np.nanmin(vtec)
vmax = np.nanmax(vtec)
# =========================
# LOOP PER JAM (EPOCH)
# =========================
for t in selected_time:

    # ambil snapshot waktu t
    mask_time = (utc == t)

    lat_h = lat[mask_time]
    lon_h = lon[mask_time]
    vtec_h = vtec[mask_time]

    if len(lat_h) == 0:
        continue

    # filter Indonesia
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
    plt.figure(figsize=(10,6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, alpha=0.2)
    ax.add_feature(cfeature.OCEAN, alpha=0.2)

    sc = ax.scatter(
    lon_id,
    lat_id,
    c=vtec_id,
    cmap='jet',
    s=8,
    vmin=vmin,
    vmax=vmax,
    transform=ccrs.PlateCarree()
)

    cbar = plt.colorbar(
    sc,
    orientation='horizontal',
    pad=0.05  # jarak dari plot
)
    cbar.set_label(r'$\mathbf{TEC\ [10^{16}/m^2]}$')
    cbar.ax.tick_params(labelsize=8)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max])

    # =========================
    # LABEL WAKTU
    # =========================
    hour_label = int(t)
    minute_label = int((t - hour_label) * 60)

    plt.title(
        f'TEC Indonesia {hour_label:02d}:{minute_label:02d}',
        fontweight='bold')

   # ========================= # SAVE # ========================= 
    plt.savefig(f'tec_{hour_label:02d}_{minute_label:02d}.png', dpi=160) 
    plt.close()