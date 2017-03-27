from os import listdir

import pandas as pd
from skimage import io, img_as_float
from skimage import transform

from constants import DM
from utils import degrade_image
from utils import framate_image

dm = 300

for f in listdir("temp"):
    print("Read file " + str(f))
    degraded = degrade_image('data/' + f, dm)
    io.imsave('temp_result/' + f, degraded)