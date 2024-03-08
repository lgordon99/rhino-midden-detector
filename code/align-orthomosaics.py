'''
align-orthomosaics by Lucia Gordon & Samuel Collier
inputs: thermal, RGB, & LiDAR tiffs
outputs: thermal, RGB, & LiDAR matrices
'''

# imports
import os
import numpy as np
import utils
from osgeo import gdal
from sys import argv

# global variables
project_dir = utils.get_project_dir()
site = utils.get_site()

# paths
thermal_tiff_path = f'{project_dir}/{site}/tiffs/thermal.tiff'
rgb_tiff_path = f'{project_dir}/{site}/tiffs/rgb.tiff'
# lidar_tiff_path = f'{folder}/tiffs/lidar.tif'

# folders
# if not os.path.exists(f'{folder}/data/'):
#     os.mkdir(f'{folder}/data/')

for modality in ['thermal', 'rgb', 'lidar']:
    # if not os.path.exists(f'{folder}/data/{modality}'):
    #     os.mkdir(f'{folder}/data/{modality}')
    os.makedirs(f'{project_dir}/{site}/data/{modality}')

# functions
def crop_array(matrix):
    start_row = 0
    end_row = matrix.shape[0]
    start_col = 0
    end_col = matrix.shape[1]

    for row_index in range(len(matrix)):
        if any(matrix[row_index] != 0):
            start_row = row_index
            break

    matrix = matrix[start_row:]

    for row_index in range(len(matrix)): 
        if all(matrix[row_index] == 0):
            end_row = row_index
            break
        else:
            end_row = matrix.shape[0]

    matrix = matrix[:end_row]
    
    for col_index in range(len(matrix.T)):
        if any(matrix.T[col_index] != 0):
            start_col = col_index
            break

    matrix = matrix.T[start_col:].T

    for col_index in range(len(matrix.T)):
        if all(matrix.T[col_index] == 0):
            end_col = col_index
            break
        else:
            end_col = matrix.shape[1]

    matrix = matrix.T[:end_col].T

    return start_row, start_row + end_row, start_col, start_col + end_col

# process thermal orthomosaic
print('thermal orthomosaic')
THERMAL_INTERVAL = 400 # width of cropped thermal images in pixels
THERMAL_STRIDE = 100 # overlap of cropped thermal images in pixels

thermal_dataset = gdal.Open(thermal_tiff_path) # converts the tiff to a Dataset object

THERMAL_NUM_ROWS = thermal_dataset.RasterYSize # pixels
THERMAL_NUM_COLS = thermal_dataset.RasterXSize # pixels

THERMAL_PIXEL_HEIGHT = thermal_dataset.GetGeoTransform()[5] # m
THERMAL_PIXEL_WIDTH = thermal_dataset.GetGeoTransform()[1] # m

THERMAL_TOP = thermal_dataset.GetGeoTransform()[3] # m
THERMAL_LEFT = thermal_dataset.GetGeoTransform()[0] # m
THERMAL_BOTTOM = THERMAL_TOP + THERMAL_PIXEL_HEIGHT * THERMAL_NUM_ROWS # m
THERMAL_RIGHT = THERMAL_LEFT + THERMAL_PIXEL_WIDTH * THERMAL_NUM_COLS # m
print('top = ' + str(THERMAL_TOP) + ', bottom = ' + str(THERMAL_BOTTOM) + ', left = ' + str(THERMAL_LEFT) + ', right = ' + str(THERMAL_RIGHT))

thermal_band = ((thermal_dataset.GetRasterBand(4)).ReadAsArray(0, 0, THERMAL_NUM_COLS, THERMAL_NUM_ROWS).astype(np.float32)) # 4th band corresponds to thermal data
THERMAL_ORTHOMOSAIC_MIN = np.amin(np.ma.masked_less(thermal_band, 2000)) # min pixel value in orthomosaic, excluding background
thermal_orthomosaic = np.ma.masked_less(thermal_band - THERMAL_ORTHOMOSAIC_MIN, 0).filled(0) # downshift the pixel values such that the min of the orthomosaic is 0 and set the backgnp.around pixels to 0
print('original orthomosaic shape =', thermal_orthomosaic.shape) # pixels

THERMAL_START_ROW, THERMAL_END_ROW, THERMAL_START_COL, THERMAL_END_COL = crop_array(thermal_orthomosaic) # extract indices for cropping
print('start row = ' + str(THERMAL_START_ROW) + ', end row = ' + str(THERMAL_END_ROW) + ', start col = ' + str(THERMAL_START_COL) + ', end col = ' + str(THERMAL_END_COL))

thermal_orthomosaic = thermal_orthomosaic[THERMAL_START_ROW : THERMAL_END_ROW, THERMAL_START_COL : THERMAL_END_COL] # crop out rows and columns that are 0
print('orthomosaic shape after cropping =', thermal_orthomosaic.shape) # pixels

new_thermal_rows = np.zeros((int(np.ceil((thermal_orthomosaic.shape[0] - THERMAL_INTERVAL) / (THERMAL_INTERVAL / 2 + THERMAL_STRIDE))) * int(THERMAL_INTERVAL / 2 + THERMAL_STRIDE) + THERMAL_INTERVAL - thermal_orthomosaic.shape[0], thermal_orthomosaic.shape[1])) # add rows so that nothing gets cut off in cropping
thermal_orthomosaic = np.vstack((thermal_orthomosaic, new_thermal_rows)) # add rows to bottom of thermal orthomosaic
new_thermal_cols = np.zeros((thermal_orthomosaic.shape[0], int(np.ceil((thermal_orthomosaic.shape[1] - THERMAL_INTERVAL) / (THERMAL_INTERVAL / 2 + THERMAL_STRIDE))) * int(THERMAL_INTERVAL / 2+THERMAL_STRIDE) + THERMAL_INTERVAL - thermal_orthomosaic.shape[1])) # add columns so that nothing gets cut off in cropping
thermal_orthomosaic = np.hstack((thermal_orthomosaic, new_thermal_cols)) # add columns to right of thermal orthomosaic
print('orthomosaic shape after adjusting size for future cropping =', thermal_orthomosaic.shape) # pixels

THERMAL_TOP_FINAL = THERMAL_TOP + THERMAL_START_ROW * THERMAL_PIXEL_HEIGHT # m
THERMAL_LEFT_FINAL = THERMAL_LEFT + THERMAL_START_COL * THERMAL_PIXEL_WIDTH # m
THERMAL_BOTTOM_FINAL = THERMAL_TOP_FINAL + THERMAL_PIXEL_HEIGHT * thermal_orthomosaic.shape[0] # m
THERMAL_RIGHT_FINAL = THERMAL_LEFT_FINAL + THERMAL_PIXEL_WIDTH * thermal_orthomosaic.shape[1] # m

print('final top = ' + str(THERMAL_TOP_FINAL) + ', final bottom = ' + str(THERMAL_BOTTOM_FINAL) + ', final left = ' + str(THERMAL_LEFT_FINAL) + ', final right = ' + str(THERMAL_RIGHT_FINAL))

constants = [['THERMAL_INTERVAL', THERMAL_INTERVAL], ['THERMAL_STRIDE', THERMAL_STRIDE], ['THERMAL_LEFT_FINAL', THERMAL_LEFT_FINAL], ['THERMAL_TOP_FINAL', THERMAL_TOP_FINAL], ['THERMAL_PIXEL_WIDTH', THERMAL_PIXEL_WIDTH], ['THERMAL_PIXEL_HEIGHT', THERMAL_PIXEL_HEIGHT], ['THERMAL_ORTHOMOSAIC_ROWS', thermal_orthomosaic.shape[0]], ['THERMAL_ORTHOMOSAIC_COLS', thermal_orthomosaic.shape[1]]]

np.save(f'{folder}/data/constants', constants)
np.save(f'{folder}/data/thermal/thermal-orthomosaic-matrix', thermal_orthomosaic) # save thermal orthomosaic as numpy array

# process RGB orthomosaic
print('RGB orthomosaic')
RGB_INTERVAL = 400 # width of cropped RGB images in pixels
RGB_STRIDE = 100 # overlap of cropped RGB images in pixels

rgb_dataset = gdal.Open(rgb_tiff_path) # converts the tiff to a Dataset object

RGB_NUM_ROWS = rgb_dataset.RasterYSize # pixels
RGB_NUM_COLS = rgb_dataset.RasterXSize # pixels
RGB_NUM_BANDS = rgb_dataset.RasterCount # 3 bands

RGB_PIXEL_HEIGHT = rgb_dataset.GetGeoTransform()[5] # m
RGB_PIXEL_WIDTH = rgb_dataset.GetGeoTransform()[1] # m

RGB_TOP = rgb_dataset.GetGeoTransform()[3] # m
RGB_LEFT = rgb_dataset.GetGeoTransform()[0] # m
RGB_BOTTOM = RGB_TOP + RGB_PIXEL_HEIGHT * RGB_NUM_ROWS # m
RGB_RIGHT = RGB_LEFT + RGB_PIXEL_WIDTH * RGB_NUM_COLS # m
print('top = ' + str(RGB_TOP) + ', bottom = ' + str(RGB_BOTTOM) + ', left = ' + str(RGB_LEFT) + ', right = ' + str(RGB_RIGHT))

rgb_bands = np.zeros((RGB_NUM_ROWS, RGB_NUM_COLS, RGB_NUM_BANDS)) # empty RGB orthomosaic

for band in range(RGB_NUM_BANDS):
    rgb_bands[:,:,band] = (rgb_dataset.GetRasterBand(band+1)).ReadAsArray(0, 0, RGB_NUM_COLS, RGB_NUM_ROWS) # add band data to RGB orthomosaic

print('original orthomosaic shape =', rgb_bands.shape)

rgb_orthomosaic = rgb_bands[int((THERMAL_TOP_FINAL - RGB_TOP) / RGB_PIXEL_HEIGHT) : int(rgb_bands.shape[0] + (THERMAL_BOTTOM_FINAL - RGB_BOTTOM) / RGB_PIXEL_HEIGHT), int((THERMAL_LEFT_FINAL - RGB_LEFT) / RGB_PIXEL_WIDTH) : int(rgb_bands.shape[1] + (THERMAL_RIGHT_FINAL - RGB_RIGHT) / RGB_PIXEL_WIDTH)].astype('uint8') # crop the RGB orthomosaic to cover the same area as the thermal orthomosaic
print('orthomosaic shape after cropping to match thermal =', rgb_orthomosaic.shape)

np.save(f'{folder}/data/rgb/rgb-orthomosaic-matrix', rgb_orthomosaic) # save RGB orthomosaic as numpy array

# process LiDAR orthomosaic
# print('LiDAR orthomosaic')
# LIDAR_INTERVAL = 80 # width of cropped LiDAR images in pixels
# LIDAR_STRIDE = 20 # overlap of cropped LiDAR images in pixels

# lidar_dataset = gdal.Open(lidar_tiff_path)

# LIDAR_NUM_ROWS = lidar_dataset.RasterYSize # pixels
# LIDAR_NUM_COLS = lidar_dataset.RasterXSize # pixels

# LIDAR_PIXEL_HEIGHT = lidar_dataset.GetGeoTransform()[5] # m
# LIDAR_PIXEL_WIDTH = lidar_dataset.GetGeoTransform()[1] # m

# LIDAR_TOP = lidar_dataset.GetGeoTransform()[3] # m
# LIDAR_LEFT = lidar_dataset.GetGeoTransform()[0] # m
# LIDAR_BOTTOM = LIDAR_TOP + LIDAR_PIXEL_HEIGHT * LIDAR_NUM_ROWS # m
# LIDAR_RIGHT = LIDAR_LEFT + LIDAR_PIXEL_WIDTH * LIDAR_NUM_COLS # m
# print('top = ' + str(LIDAR_TOP) + ', bottom = ' + str(LIDAR_BOTTOM) + ', left = ' + str(LIDAR_LEFT) + ', right = ' + str(LIDAR_RIGHT))

# lidar_band = (lidar_dataset.GetRasterBand(1)).ReadAsArray(0, 0, LIDAR_NUM_COLS, LIDAR_NUM_ROWS)
# lidar_orthomosaic_masked = np.ma.masked_equal(lidar_band, -9999).filled(0)
# print('original orthomosaic shape =', lidar_orthomosaic_masked.shape)

# lidar_orthomosaic = lidar_orthomosaic_masked[int((THERMAL_TOP_FINAL - LIDAR_TOP) / LIDAR_PIXEL_HEIGHT) : int(lidar_orthomosaic_masked.shape[0] + (THERMAL_BOTTOM_FINAL - LIDAR_BOTTOM) / LIDAR_PIXEL_HEIGHT), int((THERMAL_LEFT_FINAL - LIDAR_LEFT) / LIDAR_PIXEL_WIDTH) : int(lidar_orthomosaic_masked.shape[1] + (THERMAL_RIGHT_FINAL - LIDAR_RIGHT) / LIDAR_PIXEL_WIDTH)].astype('uint8') # crop the LiDAR orthomosaic to cover the same area as the thermal orthomosaic

# if lidar_orthomosaic.shape[0] != thermal_orthomosaic.shape[0] * LIDAR_INTERVAL / THERMAL_INTERVAL: # if the LiDAR orthomosaic has fewer rows than the thermal orthomosaic
#     new_rows = np.zeros((int(thermal_orthomosaic.shape[0] * LIDAR_INTERVAL / THERMAL_INTERVAL - lidar_orthomosaic.shape[0]), lidar_orthomosaic.shape[1]))
#     lidar_orthomosaic = np.vstack((lidar_orthomosaic, new_rows))

# print('orthomosaic shape after cropping to match thermal =', lidar_orthomosaic.shape)

# np.save(f'{folder}/data/lidar/lidar-orthomosaic-matrix', lidar_orthomosaic) # save LiDAR orthomosaic as numpy array