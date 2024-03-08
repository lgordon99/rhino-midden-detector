'''
rasterize.py by Lucia Gordon
To run, type in the terminal
python rasterize.py VECTOR_PATH RASTER_PATH RESOLUTION
VECTOR_PATH should end in .shp
RASTER_PATH should end in .tif
'''

# imports
from rasterio import features
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import sys

def rasterize(vector_path, raster_path, resolution): # resolution = number of meters covered by each pixel in the TIFF
    '''Converts a vector to a raster at a specified resolution'''

    # rasterize shapefile
    vector = gpd.read_file(vector_path)
    bounds = vector.total_bounds
    width = int((bounds[2] - bounds[0]) / resolution)
    height = int((bounds[3] - bounds[1]) / resolution)
    transform = rasterio.transform.from_origin(bounds[0], bounds[3], resolution, resolution)

    with rasterio.open(raster_path,
                       'w',
                       driver='GTiff',
                       height=height,
                       width=width,
                       count=1,
                       dtype=rasterio.uint8,
                       crs=vector.crs,
                       transform=transform) as dst:

        burned = features.rasterize(((geometry, 255) for geometry in vector.geometry),
                                     out_shape=(height, width),
                                     transform=transform,
                                     fill=0,
                                     all_touched=True,
                                     dtype=rasterio.uint8) # burn the features into the raster

        dst.write_band(1, burned) # write the rasterized shapefile to the GeoTIFF

    # convert raster to array
    array = rasterio.open(raster_path).read(1) # 0 = not feature, 255 = feature
    print(f'Array shape = {array.shape}')
    print(f'Min array = {np.amin(array)}, max array = {np.amax(array)}')
    print(f'No feature value = {array[0,0]}')

    # check that all values are 0 or 255
    for row in range(array.shape[0]):
        for col in range(array.shape[1]):
            if array[row][col] != 0 and array[row][col] != 255:
                print(array[row][col])

    # plot raster
    plt.figure(dpi=300)
    plt.imshow(array) # plot the array of pixel values as an image
    plt.axis('off') # remove axes
    plt.savefig(f'raster_{resolution}m.png', bbox_inches='tight', pad_inches=0)
    plt.close() # close the image to save memory

    print(f'Rasterized at {resolution}m resolution')

if __name__ == '__main__':
    rasterize(vector_path=sys.argv[1], raster_path=sys.argv[2], resolution=float(sys.argv[3]))
