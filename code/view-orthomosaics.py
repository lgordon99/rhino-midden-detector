'''view-orthomosaics by Lucia Gordon'''

# imports
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sys import argv

# global variables
folder = argv[1]

# folders
if not os.path.exists(f'{folder}/png-images'):
    os.mkdir(f'{folder}/png-images')

if not os.path.exists(f'{folder}/png-images/orthomosaics'):
    os.mkdir(f'{folder}/png-images/orthomosaics')

# functions
def plot_orthomosaic(modality):
    plt.figure(dpi = 300)
    image = plt.imshow(np.load(f'{folder}/data/{modality}/{modality}-orthomosaic-matrix.npy'))

    if modality == 'thermal' or modality == 'lidar':
        image.set_cmap('inferno')

    plt.axis('off') # remove axes        
    plt.savefig(f'{folder}/png-images/orthomosaics/{modality}-orthomosaic.png', bbox_inches = 'tight', pad_inches = 0) # temporarily save the image
    plt.close() # close the image to save memory

# plot orthomosaics
plot_orthomosaic('thermal')
print('thermal orthomosaic plotted')

plot_orthomosaic('rgb')
print('RGB orthomosaic plotted')

# plot_orthomosaic('lidar')
# print('LiDAR orthomosaic plotted')

# fuse orthomosaics
thermal_orthomosaic = Image.open(f'{folder}/png-images/orthomosaics/thermal-orthomosaic.png')
rgb_orthomosaic = Image.open(f'{folder}/png-images/orthomosaics/rgb-orthomosaic.png')
# lidar_orthomosaic = Image.open(f'{folder}/png-images/orthomosaics/lidar-orthomosaic.png')
thermal_rgb_fused = Image.blend(thermal_orthomosaic, rgb_orthomosaic, 0.5)
thermal_rgb_fused.save(f'{folder}/png-images/orthomosaics/thermal_rgb_fused.png', 'PNG')
print('thermal and RGB orthomosaics have been fused')

# thermal_rgb_lidar_fused = Image.blend(thermal_rgb_fused, lidar_orthomosaic, 1/3)
# thermal_rgb_lidar_fused.save(f'{folder}/png-images/orthomosaics/thermal_rgb_lidar_fused.png', 'PNG')
# print('thermal, RGB, and LiDAR orthomosaics have been fused')