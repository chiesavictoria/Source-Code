import imageio.v2 as imageio
import glob
import os

files = sorted(
    glob.glob("GNSS TEC/TEC_05MAY_10MINS/*.png"),
    key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0])
)

# cek urutan (opsional tapi penting)
print("\n".join(files[:5]))
print("...")
print("\n".join(files[-5:]))

images = [imageio.imread(f) for f in files]

imageio.mimsave(
    "TECGNSS_05May2024_10MINS.mp4",
    images,
    fps=2
)