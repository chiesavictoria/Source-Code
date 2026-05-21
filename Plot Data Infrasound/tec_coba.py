import imageio.v2 as imageio
import glob
import os

files = sorted(
    glob.glob("Plot Data Infrasound/Thailand/11May2024/*.png"),
    key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0])
)

# cek urutan (opsional tapi penting)
print("\n".join(files[:5]))
print("...")
print("\n".join(files[-5:]))

images = [imageio.imread(f) for f in files]

imageio.mimsave(
    "IonosondeData_11May2024.gif",
    images,
    duration=8
)