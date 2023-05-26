'''
process-data by Lucia Gordon
inputs: thermal, RGB, LiDAR, and midden matrices
outputs: thermal, RGB, and LiDAR PNG images and arrays, identifiers, labels, and thermal maximum pixel values
'''

# imports
import os
import numpy as np
import matplotlib.pyplot as plt
from shutil import rmtree
from cv2 import imread

# folders
if os.path.exists('png-images'):
    rmtree('png-images')

os.mkdir('png-images')

for modality in ['thermal', 'rgb', 'lidar']:
    os.mkdir('png-images/'+modality)

# functions
def save_png_matrices(raw_images, modality):
    png_arrays = []

    for i in range(len(raw_images)):
        plt.figure(dpi=60.7) # dpi=60.7 to get resultant arrays of (224,224,3), dpi=11 to get resultant arrays of (40,40,3), 27.3 for 100
        image = plt.imshow(raw_images[i]) # plot the array of pixel values as an image

        if modality == 'thermal' or modality == 'lidar':
            image.set_cmap('inferno')
        
        plt.axis('off') # remove axes        
        plt.savefig('png-images/'+modality+'/'+modality+'-'+str(i)+'.png', bbox_inches='tight', pad_inches=0) # temporarily save the image
        plt.close() # close the image to save memory
        png_arrays.append(imread('png-images/'+modality+'/'+modality+'-'+str(i)+'.png')) # convert the PNG image to a 3D array
    
    np.save('data/'+modality+'/'+modality+'-images', png_arrays)

# crop thermal orthomosaic
thermal_orthomosaic = np.load('data/thermal/thermal-orthomosaic-matrix.npy')
midden_matrix = np.load('data/midden-matrix.npy')
THERMAL_INTERVAL = 40 # width and height of cropped thermal images in pixels
THERMAL_STRIDE = 10 # overlap of cropped thermal images in pixels
THERMAL_STEP = int(THERMAL_INTERVAL/2 + THERMAL_STRIDE) # 30
thermal_images = [] # images cropped from orthomosaic
thermal_label_matrices = [] # midden locations cropped from orthomosaic
thermal_midden_images = [] # subset of the cropped images that contain middens
thermal_empty_images = [] # subset of the cropped images that are empty

num_vertical = 0

for bottom in range(THERMAL_INTERVAL, thermal_orthomosaic.shape[0], THERMAL_STEP): # begin cropping from the top of the orthomosaic
    num_horizontal = 0

    for right in range(THERMAL_INTERVAL, thermal_orthomosaic.shape[1], THERMAL_STEP): # begin cropping from the left end of the orthomosaic
        cropped_image = thermal_orthomosaic[bottom-THERMAL_INTERVAL:bottom,right-THERMAL_INTERVAL:right].copy() # create an image cropped from the orthomosaic
        cropped_image -= np.amin(cropped_image) # set the minimum pixel value to 0
        thermal_images.append(cropped_image) # save cropped image to list
        thermal_label_matrices.append(midden_matrix[bottom-THERMAL_INTERVAL:bottom,right-THERMAL_INTERVAL:right]) # save the same cropping from the matrix of midden locations
        num_horizontal += 1

    num_vertical += 1

constants = list(np.load('data/constants.npy'))
constants.append(['NUM_HORIZONTAL', num_horizontal])
labels = list(np.sum(np.sum(thermal_label_matrices, axis=1), axis=1)) # collapses each label matrix to the number of middens in the corresponding cropped image
identifiers = list(range(len(thermal_images)))
max_pixel_vals = [np.amax(thermal_images[i]) for i in range(len(thermal_images))]

for index in range(len(labels)):
    if labels[index] > 1: # if there happens to be more than 1 midden in an image
        labels[index] = 1 # set the label to 1 since we only care about whether there is a midden or not

# crop RGB orthomosaic
rgb_orthomosaic = np.load('data/rgb/rgb-orthomosaic-matrix.npy')
RGB_INTERVAL = 400 # width of cropped thermal images in pixels
RGB_STRIDE = 100 # overlap of cropped thermal images in pixels
RGB_STEP = int(RGB_INTERVAL/2 + RGB_STRIDE) # 300
rgb_images = [] # images cropped from orthomosaic
rgb_midden_images = [] # subset of the cropped images that contain middens
rgb_empty_images = [] # subset of the cropped images that are empty

for bottom in range(RGB_INTERVAL, rgb_orthomosaic.shape[0], RGB_STEP): # begin cropping from the top of the orthomosaic
    for right in range(RGB_INTERVAL, rgb_orthomosaic.shape[1], RGB_STEP): # begin cropping from the left end of the orthomosaic
        cropped_image = rgb_orthomosaic[bottom-RGB_INTERVAL:bottom,right-RGB_INTERVAL:right].copy() # create an image cropped from the orthomosaic
        rgb_images.append(cropped_image) # save cropped image to list

# crop LiDAR orthomosaic
lidar_orthomosaic = np.load('data/lidar/lidar-orthomosaic-matrix.npy')
LIDAR_INTERVAL = 80
LIDAR_STRIDE = 20
LIDAR_STEP = int(LIDAR_INTERVAL/2 + LIDAR_STRIDE) # 60
lidar_images = []
lidar_midden_images = []
lidar_empty_images = []

for bottom in range(LIDAR_INTERVAL, lidar_orthomosaic.shape[0], LIDAR_STEP): # begin cropping from the top of the orthomosaic
    for right in range(LIDAR_INTERVAL, lidar_orthomosaic.shape[1], LIDAR_STEP): # begin cropping from the left end of the orthomosaic
        cropped_image = lidar_orthomosaic[bottom-LIDAR_INTERVAL:bottom,right-LIDAR_INTERVAL:right].copy() # create an image cropped from the orthomosaic
        lidar_images.append(cropped_image) # save cropped image to list

# remove empty images
for i in reversed(range(len(labels))):
    if np.all(thermal_images[i]==0) or np.all(rgb_images[i]==0):
        del thermal_images[i]
        del rgb_images[i]
        del lidar_images[i]
        del max_pixel_vals[i]
        del identifiers[i]
        del labels[i]

print(len(thermal_images))
print(len(rgb_images))
print(len(lidar_images))
print(len(max_pixel_vals))
print(len(identifiers))
print(len(labels))

# save data
save_png_matrices(thermal_images, 'thermal')
print('thermal images saved')
save_png_matrices(rgb_images, 'rgb')
print('RGB images saved')
save_png_matrices(lidar_images, 'lidar')
print('LiDAR images saved')
np.save('data/constants', constants)
np.save('data/labels', labels)
np.save('data/identifiers', identifiers)
np.save('data/max-pixel-vals', max_pixel_vals)