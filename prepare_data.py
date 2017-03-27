from os import listdir

import pandas as pd
from skimage import io, img_as_float
from skimage import transform

from constants import DM
from utils import degrade_image
from utils import framate_image


def read_frame(file, s):
    img = img_as_float(io.imread(file))
    return framate_image(img, DM, s)


for f in listdir("data"):
    print("Read file " + str(f))
    degraded = degrade_image('data/' + f, DM)
    io.imsave('resized/' + f, degraded)
    io.imsave('resized/over_' + f, transform.rotate(degraded, 180, resize=False))
    io.imsave('resized_90/' + f, transform.rotate(degraded, 90, resize=False))
    io.imsave('resized_90/' + f, transform.rotate(degraded, 270, resize=False))

df_all = pd.DataFrame()
for d in ["resized"]:
    for f in listdir(d):
        print("make frame from " + str(f))
        df_all = df_all.append(read_frame(d + '/' + f, 1))

for d in ["resized_90"]:
    for f in listdir(d):
        print("make frame from " + str(f))
        df_all = df_all.append(read_frame(d + '/' + f, 0))

print("Prepare file with features")
df_all.to_csv("deatures.csv", index=False)
