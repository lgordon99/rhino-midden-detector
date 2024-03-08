'''
fuse by Lucia Gordon
inputs: thermal, RGB, and LiDAR image arrays
outputs: fused image arrays for thermal-RGB, thermal-LiDAR, and RGB-LiDAR
'''

# imports
import os
import numpy as np
from PIL import Image
from shutil import rmtree
from cv2 import imread
from sys import argv

# global variables
folder = argv[1]
thermal_images = np.load(folder+'/data/thermal/thermal-images.npy')
rgb_images = np.load(folder+'/data/rgb/rgb-images.npy')
lidar_images = np.load(folder+'/data/lidar/lidar-images.npy')

# folders
if os.path.exists(folder+'/data/fused'):
    rmtree(folder+'/data/fused')

os.mkdir(folder+'/data/fused')

# functions
def fuse(modality1, images1, modality2, images2, modality3=None, images3=None):
    fused_arrays = []

    for i in range(len(images1)):
        image1 = Image.open('png-images/'+modality1+'/'+modality1+'-'+str(i)+'.png')
        image2 = Image.open('png-images/'+modality2+'/'+modality2+'-'+str(i)+'.png')
        fused = Image.blend(image1, image2, 0.5)
        fused.save('fused.png', 'PNG')

        if modality3 is not None:
            image3 = Image.open('png-images/'+modality3+'/'+modality3+'-'+str(i)+'.png')
            fused = Image.blend(fused, image3, 1/3)
            fused.save('fused.png', 'PNG')
        
        fused_arrays.append(imread('fused.png'))

    return fused_arrays

# generate and save fused image arrays
np.save(folder+'/data/fused/tr-fused', fuse('thermal', thermal_images, 'rgb', rgb_images))
print('thermal-RGB fusing done')
np.save(folder+'/data/fused/tl-fused', fuse('thermal', thermal_images, 'lidar', lidar_images))
print('thermal-LiDAR fusing done')
np.save(folder+'/data/fused/rl-fused', fuse('rgb', rgb_images, 'lidar', lidar_images))
print('RGB-LiDAR fusing done')
np.save(folder+'/data/fused/trl-fused', fuse('thermal', thermal_images, 'rgb', rgb_images, 'lidar', lidar_images))
print('thermal-RGB-LiDAR fusing done')

# remove temporary image
os.remove('fused.png')