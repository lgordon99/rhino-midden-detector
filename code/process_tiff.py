'''
process_tiff.py by Lucia Gordon
To run, type in the terminal
python process_tiff.py TIFF_PATH
TIFF_PATH should end in .tif
OR
python process_tiff.py TIFF_PATH PNG_PATH
PNG_PATH should end in .png
'''

# imports
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import sys

def read_tiff(tiff_path):
    '''Reads the data in a TIFF file at a given path'''

    tiff = rasterio.open(tiff_path)
    array = tiff.read()
    print(f'tiff array shape = {array.shape}')
    width, height = tiff.meta['width'], tiff.meta['height'] # pixels
    print(f'width = {width}, height = {height}')
    num_bands = tiff.meta['count']
    print(f'{num_bands} band(s)')
    x_res, y_res = tiff.meta['transform'][0], tiff.meta['transform'][4] # meters per pixel
    print(f'horizontal resolution = {x_res}, vertical resolution = {y_res}')
    left, bottom, right, top = tiff.bounds # meters
    print(f'left = {left}, bottom = {bottom}, right = {right}, top = {top}')

def plot_tiff(tiff_path, png_path):
    '''Converts a TIFF to a PNG'''

    array = rasterio.open(tiff_path).read(1) # use "1" as the argument in .read() for LiDAR and "4" for thermal

    plt.figure(dpi=300)
    plt.imshow(array) # plot the array of pixel values as an image
    plt.axis('off') # remove axes        
    plt.savefig(png_path, bbox_inches='tight', pad_inches=0)
    plt.close() # close the image to save memory

if __name__ == '__main__':
    read_tiff(tiff_path=sys.argv[1])

    if len(sys.argv) > 2:
        plot_tiff(tiff_path=sys.argv[1], png_path=sys.argv[2])
