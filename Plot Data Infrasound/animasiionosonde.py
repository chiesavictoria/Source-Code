import imageio.v2 as imageio
import glob
import os

files = sorted(
    glob.glob("Plot Data Infrasound/TEC/*.jpg"),
    key=lambda x: int(os.path.basename(x).split("_")[1])
)

images = [imageio.imread(f) for f in files]

imageio.mimsave("TEC_animation2.gif", images, duration=0.5)