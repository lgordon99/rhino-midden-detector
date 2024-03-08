# imports
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import utils
from random import sample

project_dir = utils.get_project_dir()
site = utils.get_site()
site_dir = f'{project_dir}/{site}'
identifiers = np.load(f'{site_dir}/data/identifiers.npy')

batch = 1
os.makedirs(f'{site_dir}/batch-{batch}-images', exist_ok=True)
batch_identifiers = sample(list(identifiers), k=1)

def plot_image_from_id(identifier, modality):
    index = np.where(identifiers == identifier)[0][0]

    plt.figure(dpi=300)
    image = plt.imshow(np.load(f'{site_dir}/data/{modality}/{modality}-images.npy')[index]) # plot the array of pixel values as an image

    if modality == 'thermal' or modality == 'lidar':
        image.set_cmap('inferno')
    plt.colorbar()
    plt.axis('off') # remove axes  
    plt.savefig(f'{site_dir}/batch-{batch}-images/img-{identifier}-{modality[0]}.png', bbox_inches='tight', pad_inches=0)
    plt.close() # close the image to save memory

def plot_zoomed_out_image_from_id(identifier, modality):
    THERMAL_INTERVAL = utils.get_thermal_interval()
    x_pixels, y_pixels = utils.get_image_center_pixels(identifier)
    x, y = utils.get_image_center_meters(x_pixels, y_pixels)
    orthomosaic = np.load(f'{site_dir}/data/{modality}/{modality}-orthomosaic-matrix.npy')
    top_row = int(y_pixels - THERMAL_INTERVAL/2)
    bottom_row = int(y_pixels + THERMAL_INTERVAL/2)
    left_column = int(x_pixels - THERMAL_INTERVAL/2)
    right_column = int(x_pixels + THERMAL_INTERVAL/2)
    
    # put a box around the image in question
    for i in range(top_row, bottom_row+1):
        for j in range(left_column, right_column+1):
            if (i == top_row or i == bottom_row) or (j == left_column or j == right_column):
                orthomosaic[i][j] = 0

    zoomed_out_array = orthomosaic[max(top_row-THERMAL_INTERVAL, 0) : min(bottom_row+THERMAL_INTERVAL, len(orthomosaic)), max(left_column-THERMAL_INTERVAL, 0) : min(right_column+THERMAL_INTERVAL, orthomosaic.shape[1])]

    plt.figure(dpi=300)
    zoomed_out_image = plt.imshow(zoomed_out_array)

    if modality == 'thermal' or modality == 'lidar':
        zoomed_out_image.set_cmap('inferno')
    plt.colorbar()
    plt.axis('off') # remove axes
    plt.savefig(f'{site_dir}/batch-{batch}-images/img-{identifier}-{modality[0]}-zoomed-out.png', bbox_inches='tight', pad_inches=0)
    plt.close() # close the image to save memory

for identifier in batch_identifiers:
    plot_image_from_id(identifier, 'thermal')
    # plot_image_from_id(identifier, 'rgb')
    plot_zoomed_out_image_from_id(identifier, 'thermal')
    # plot_zoomed_out_image_from_id(identifier, 'rgb')
