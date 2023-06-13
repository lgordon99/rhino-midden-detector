'''view-images by Lucia Gordon'''

# imports
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shutil import rmtree

# global variables
constants = np.load('data/constants.npy')
THERMAL_INTERVAL = int(constants[0][1])
THERMAL_STRIDE = int(constants[1][1])
THERMAL_LEFT = float(constants[2][1])
THERMAL_TOP = float(constants[3][1])
THERMAL_PIXEL_WIDTH = float(constants[4][1])
THERMAL_PIXEL_HEIGHT = float(constants[5][1])
NUM_HORIZONTAL = int(constants[6][1])
labels = np.load('data/labels.npy')
identifiers = np.load('data/identifiers.npy')

# folders
if os.path.exists('image-test'):
    rmtree('image-test')

os.mkdir('image-test')

# functions
def get_image_center_pixels(identifier):
    row = np.floor(identifier/NUM_HORIZONTAL)
    col = identifier - NUM_HORIZONTAL*np.floor(identifier/NUM_HORIZONTAL)
    x_pixels = col*(THERMAL_STRIDE+THERMAL_INTERVAL/2) + THERMAL_INTERVAL/2
    y_pixels = row*(THERMAL_STRIDE+THERMAL_INTERVAL/2) + THERMAL_INTERVAL/2

    return x_pixels, y_pixels

def get_image_center_meters(x_pixels, y_pixels):
    x = THERMAL_LEFT + x_pixels*THERMAL_PIXEL_WIDTH
    y = THERMAL_TOP + y_pixels*THERMAL_PIXEL_HEIGHT

    return x, y

# select image
index = random.choice(list(np.where(labels == 1)[0]))
print('index =', index)
identifier = identifiers[index]
print('identifier =', identifier)

# show selected image based on its index
for modality in ['thermal', 'rgb', 'lidar']:
    plt.figure(dpi=300)
    image = plt.imshow(np.load('data/'+modality+'/'+modality+'-images.npy')[index]) # plot the array of pixel values as an image

    if modality == 'thermal' or modality == 'lidar':
        image.set_cmap('inferno')

    plt.axis('off') # remove axes        
    plt.savefig('image-test/img'+str(identifier)+'-'+modality[0]+'.png', bbox_inches='tight', pad_inches=0) # temporarily save the image
    plt.close() # close the image to save memory

# get coordinates of the center of the selected image
x_pixels, y_pixels = get_image_center_pixels(identifier)
print('image center in pixels: x = ' + str(x_pixels) + ', y = ' + str(y_pixels))
x, y = get_image_center_meters(x_pixels, y_pixels)
print('image center in meters: x = ' + str(x) + ' m, y = ' + str(y) + ' m')

# show selected image based on its coordinates
thermal_orthomosaic = np.load('data/thermal/thermal-orthomosaic-matrix.npy')
im = thermal_orthomosaic[int(y_pixels-THERMAL_INTERVAL/2):int(y_pixels+THERMAL_INTERVAL/2),int(x_pixels-THERMAL_INTERVAL/2):int(x_pixels+THERMAL_INTERVAL/2)].copy()
plt.figure(dpi=300)
image = plt.imshow(im) # plot the array of pixel values as an image
image.set_cmap('inferno')
plt.axis('off') # remove axes        
plt.savefig('image-test/img'+str(identifier)+'-rawthermal.png', bbox_inches='tight', pad_inches=0) # temporarily save the image
plt.close() # close the image to save memory

# plot the location of the image relative to the middens in the area
midden_coords = pd.read_csv('data/midden-coordinates-m.csv').to_numpy().T # in meters
plt.figure(dpi=300)
plt.scatter(midden_coords[0], midden_coords[1], s=20, c='b')
plt.scatter(x, y, s=10, c='r')
plt.savefig('image-test/midden-plot.png', bbox_inches='tight')
plt.close()