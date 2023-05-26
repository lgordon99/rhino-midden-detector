'''
align-orthomosaics-and-middens by Lucia Gordon & Samuel Collier
inputs: thermal, RGB, and LiDAR tiffs and the midden coordinates
outputs: thermal, RGB, LiDAR, and midden matrices
'''

# imports
import os
import numpy as np
import pandas as pd
from osgeo import gdal
from shutil import rmtree
from sys import argv

# paths (change as needed)
thermal_tiff_path = 'tiffs/thermal-50cm.tif'
rgb_tiff_path = 'tiffs/rgb-5cm.tif' 
lidar_tiff_path = 'tiffs/lidar-dtm25cm.tif'
midden_coordinates_path = 'data/midden-coordinates-m.csv'

# folders
for modality in ['thermal', 'rgb', 'lidar']:
    if os.path.exists('data/'+modality):
        rmtree('data/'+modality)

    os.mkdir('data/'+modality)

# functions
def crop_array(matrix):
    start_row = 0
    end_row = matrix.shape[0]
    start_col = 0
    end_col = matrix.shape[1]

    for row_index in range(len(matrix)):
        if any(matrix[row_index]!=0):
            start_row = row_index
            break

    matrix = matrix[start_row:]

    for row_index in range(len(matrix)): 
        if all(matrix[row_index]==0):
            end_row = row_index
            break
        else:
            end_row = matrix.shape[0]

    matrix = matrix[:end_row]
    
    for col_index in range(len(matrix.T)):
        if any(matrix.T[col_index]!=0):
            start_col = col_index
            break

    matrix = matrix.T[start_col:].T

    for col_index in range(len(matrix.T)):
        if all(matrix.T[col_index]==0):
            end_col = col_index
            break
        else:
            end_col = matrix.shape[1]

    matrix = matrix.T[:end_col].T

    return start_row, start_row+end_row, start_col, start_col+end_col

# process thermal orthomosaic
THERMAL_INTERVAL = 40 # width of cropped thermal images in pixels
THERMAL_STRIDE = 10 # overlap of cropped thermal images in pixels
thermal_dataset = gdal.Open(thermal_tiff_path) # converts the tiff to a Dataset object
THERMAL_NUM_ROWS = thermal_dataset.RasterYSize # 4400 pixels
THERMAL_NUM_COLS = thermal_dataset.RasterXSize # 4400 pixels
THERMAL_TOP = thermal_dataset.GetGeoTransform()[3]
THERMAL_PIXEL_HEIGHT = thermal_dataset.GetGeoTransform()[5] # -0.5 m
THERMAL_LEFT = thermal_dataset.GetGeoTransform()[0]
THERMAL_PIXEL_WIDTH = thermal_dataset.GetGeoTransform()[1] # 0.5 m
THERMAL_BOTTOM = THERMAL_TOP + THERMAL_PIXEL_HEIGHT * THERMAL_NUM_ROWS
THERMAL_RIGHT = THERMAL_LEFT + THERMAL_PIXEL_WIDTH * THERMAL_NUM_COLS
thermal_band = ((thermal_dataset.GetRasterBand(4)).ReadAsArray(0,0,THERMAL_NUM_COLS,THERMAL_NUM_ROWS).astype(np.float32)) # 4th band corresponds to thermal data
THERMAL_ORTHOMOSAIC_MIN = np.amin(np.ma.masked_less(thermal_band,2000)) # 7638 = min pixel value in orthomosaic, excluding background
thermal_orthomosaic = np.ma.masked_less(thermal_band-THERMAL_ORTHOMOSAIC_MIN,0).filled(0) # downshift the pixel values such that the min of the orthomosaic is 0 and set the backgnp.around pixels to 0
print(thermal_orthomosaic.shape) # (4400,4400)
THERMAL_START_ROW, THERMAL_END_ROW, THERMAL_START_COL, THERMAL_END_COL = crop_array(thermal_orthomosaic) # extract indices for cropping
print(THERMAL_START_ROW, THERMAL_END_ROW, THERMAL_START_COL, THERMAL_END_COL) # (495, 4400, 339, 3716)
thermal_orthomosaic = thermal_orthomosaic[THERMAL_START_ROW:THERMAL_END_ROW,THERMAL_START_COL:THERMAL_END_COL] # crop out rows and columns that are 0
print(thermal_orthomosaic.shape) # (3905,3377)
new_thermal_rows = np.zeros((int(np.ceil((thermal_orthomosaic.shape[0]-THERMAL_INTERVAL)/(THERMAL_INTERVAL/2+THERMAL_STRIDE)))*int(THERMAL_INTERVAL/2+THERMAL_STRIDE)+THERMAL_INTERVAL-thermal_orthomosaic.shape[0],thermal_orthomosaic.shape[1])) # add rows so that nothing gets cut off in cropping
thermal_orthomosaic = np.vstack((thermal_orthomosaic, new_thermal_rows)) # add rows to bottom of thermal orthomosaic
new_thermal_cols = np.zeros((thermal_orthomosaic.shape[0],int(np.ceil((thermal_orthomosaic.shape[1]-THERMAL_INTERVAL)/(THERMAL_INTERVAL/2+THERMAL_STRIDE)))*int(THERMAL_INTERVAL/2+THERMAL_STRIDE)+THERMAL_INTERVAL-thermal_orthomosaic.shape[1])) # add columns so that nothing gets cut off in cropping
thermal_orthomosaic = np.hstack((thermal_orthomosaic, new_thermal_cols)) # add columns to right of thermal orthomosaic
print(thermal_orthomosaic.shape) # (3910,3400)
THERMAL_LEFT_FINAL = THERMAL_LEFT + THERMAL_START_COL * THERMAL_PIXEL_WIDTH
THERMAL_TOP_FINAL = THERMAL_TOP + THERMAL_START_ROW * THERMAL_PIXEL_HEIGHT
constants = [['THERMAL_INTERVAL', THERMAL_INTERVAL], ['THERMAL_STRIDE', THERMAL_STRIDE], ['THERMAL_LEFT', THERMAL_LEFT_FINAL], ['THERMAL_TOP', THERMAL_TOP_FINAL], ['THERMAL_PIXEL_WIDTH', THERMAL_PIXEL_WIDTH], ['THERMAL_PIXEL_HEIGHT', THERMAL_PIXEL_HEIGHT]]
np.save('data/constants', constants)
np.save('data/thermal/thermal-orthomosaic-matrix', thermal_orthomosaic) # save thermal orthomosaic as numpy array

# process RGB orthomosaic
RGB_INTERVAL = 400 # width of cropped RGB images in pixels
RGB_STRIDE = 100 # overlap of cropped RGB images in pixels
rgb_dataset = gdal.Open(rgb_tiff_path) # converts the tiff to a Dataset object
RGB_NUM_ROWS = rgb_dataset.RasterYSize # 54000 pixels
RGB_NUM_COLS = rgb_dataset.RasterXSize # 54000 pixels
RGB_NUM_BANDS = rgb_dataset.RasterCount # 3 bands
RGB_TOP = rgb_dataset.GetGeoTransform()[3]
RGB_PIXEL_HEIGHT = rgb_dataset.GetGeoTransform()[5] # -0.05 m
RGB_LEFT = rgb_dataset.GetGeoTransform()[0]
RGB_PIXEL_WIDTH = rgb_dataset.GetGeoTransform()[1] # 0.05 m
RGB_BOTTOM = RGB_TOP + RGB_PIXEL_HEIGHT * RGB_NUM_ROWS
RGB_RIGHT = RGB_LEFT + RGB_PIXEL_WIDTH * RGB_NUM_COLS
rgb_bands = np.zeros((RGB_NUM_ROWS,RGB_NUM_COLS,RGB_NUM_BANDS)) # empty RGB orthomosaic

for band in range(RGB_NUM_BANDS):
    rgb_bands[:,:,band] = (rgb_dataset.GetRasterBand(band+1)).ReadAsArray(0,0,RGB_NUM_COLS,RGB_NUM_ROWS) # add band data to RGB orthomosaic

rgb_orthomosaic = rgb_bands[int((THERMAL_TOP-RGB_TOP)/RGB_PIXEL_HEIGHT):int(rgb_bands.shape[0]+(THERMAL_BOTTOM-RGB_BOTTOM)/RGB_PIXEL_HEIGHT),int((THERMAL_LEFT-RGB_LEFT)/RGB_PIXEL_WIDTH):int(rgb_bands.shape[1]+(THERMAL_RIGHT-RGB_RIGHT)/RGB_PIXEL_WIDTH)].astype('uint8') # crop the RGB orthomosaic to cover the same area as the thermal orthomosaic
print(rgb_orthomosaic.shape) # (44000,44000,3)
rgb_orthomosaic = rgb_orthomosaic[10*THERMAL_START_ROW:10*THERMAL_END_ROW,10*THERMAL_START_COL:10*THERMAL_END_COL] # crop RGB orthomosaic to cover the same area as the thermal orthomosaic after removing empty rows and columns
print(rgb_orthomosaic.shape) # (39050,33770,3)
new_rgb_rows = np.zeros((int(np.ceil((rgb_orthomosaic.shape[0]-RGB_INTERVAL)/(RGB_INTERVAL/2+RGB_STRIDE)))*int(RGB_INTERVAL/2+RGB_STRIDE)+RGB_INTERVAL-rgb_orthomosaic.shape[0],rgb_orthomosaic.shape[1],rgb_orthomosaic.shape[2])) # add rows so that nothing gets cut off in cropping
rgb_orthomosaic = np.vstack((rgb_orthomosaic, new_rgb_rows)) # add rows to bottom of RGB orthomosaic
new_rgb_cols = np.zeros((rgb_orthomosaic.shape[0],int(np.ceil((rgb_orthomosaic.shape[1]-RGB_INTERVAL)/(RGB_INTERVAL/2+RGB_STRIDE)))*int(RGB_INTERVAL/2+RGB_STRIDE)+RGB_INTERVAL-rgb_orthomosaic.shape[1],rgb_orthomosaic.shape[2])) # add columns so that nothing gets cut off in cropping
rgb_orthomosaic = np.hstack((rgb_orthomosaic, new_rgb_cols)).astype('uint8') # add columns to right of RGB orthomosaic
print(rgb_orthomosaic.shape) # (39100,34000,3)
np.save('data/rgb/rgb-orthomosaic-matrix', rgb_orthomosaic) # save RGB orthomosaic as numpy array

# process LiDAR orthomosaic
LIDAR_INTERVAL = 80
LIDAR_STRIDE = 20
lidar_dataset = gdal.Open(lidar_tiff_path)
LIDAR_NUM_ROWS = lidar_dataset.RasterYSize # 10117 pixels
LIDAR_NUM_COLS = lidar_dataset.RasterXSize # 11769 pixels
LIDAR_TOP = lidar_dataset.GetGeoTransform()[3]
LIDAR_PIXEL_HEIGHT = lidar_dataset.GetGeoTransform()[5] # -0.25 m
LIDAR_LEFT = lidar_dataset.GetGeoTransform()[0]
LIDAR_PIXEL_WIDTH = lidar_dataset.GetGeoTransform()[1] # 0.25 m
LIDAR_BOTTOM = LIDAR_TOP + LIDAR_PIXEL_HEIGHT * LIDAR_NUM_ROWS
LIDAR_RIGHT = LIDAR_LEFT + LIDAR_PIXEL_WIDTH * LIDAR_NUM_COLS
lidar_band = (lidar_dataset.GetRasterBand(1)).ReadAsArray(0,0,LIDAR_NUM_COLS,LIDAR_NUM_ROWS)
lidar_orthomosaic = np.ma.masked_equal(lidar_band,-9999).filled(0)
lidar_orthomosaic = np.vstack((np.zeros((int((LIDAR_TOP-THERMAL_TOP)/LIDAR_PIXEL_HEIGHT), lidar_orthomosaic.shape[1])), lidar_orthomosaic)) # add rows to the top because the LiDAR orthomosaic is smaller than the thermal one
lidar_orthomosaic = lidar_orthomosaic[:int(lidar_orthomosaic.shape[0]+(THERMAL_BOTTOM-LIDAR_BOTTOM)/LIDAR_PIXEL_HEIGHT),int((THERMAL_LEFT-LIDAR_LEFT)/LIDAR_PIXEL_WIDTH):int(lidar_orthomosaic.shape[1]+(THERMAL_RIGHT-LIDAR_RIGHT)/LIDAR_PIXEL_WIDTH)] # crop the LiDAR orthomosaic to cover the same area as the thermal orthomosaic
print(lidar_orthomosaic.shape) # (8800,8800)
lidar_orthomosaic = lidar_orthomosaic[2*THERMAL_START_ROW:2*THERMAL_END_ROW,2*THERMAL_START_COL:2*THERMAL_END_COL] # crop LiDAR orthomosaic to cover the same area as the thermal orthomosaic after removing empty rows and columns
print(lidar_orthomosaic.shape) # (7810,6754)
new_lidar_rows = np.zeros((int(np.ceil((lidar_orthomosaic.shape[0]-LIDAR_INTERVAL)/(LIDAR_INTERVAL/2+LIDAR_STRIDE)))*int(LIDAR_INTERVAL/2+LIDAR_STRIDE)+LIDAR_INTERVAL-lidar_orthomosaic.shape[0],lidar_orthomosaic.shape[1])) # add rows so that nothing gets cut off in cropping
lidar_orthomosaic = np.vstack((lidar_orthomosaic, new_lidar_rows)) # add rows to bottom of LiDAR orthomosaic
new_lidar_cols = np.zeros((lidar_orthomosaic.shape[0],int(np.ceil((lidar_orthomosaic.shape[1]-LIDAR_INTERVAL)/(LIDAR_INTERVAL/2+LIDAR_STRIDE)))*int(LIDAR_INTERVAL/2+LIDAR_STRIDE)+LIDAR_INTERVAL-lidar_orthomosaic.shape[1])) # add columns so that nothing gets cut off in cropping
lidar_orthomosaic = np.hstack((lidar_orthomosaic, new_lidar_cols)) # add columns to right of LiDAR orthomosaic
print(lidar_orthomosaic.shape) # (7820,6800)
np.save('data/lidar/lidar-orthomosaic-matrix', lidar_orthomosaic) # save LiDAR orthomosaic as numpy array

# process middens
midden_coords = pd.read_csv(midden_coordinates_path).to_numpy().T # in meters
midden_coords[0] = (midden_coords[0]-THERMAL_LEFT)/THERMAL_PIXEL_WIDTH - THERMAL_START_COL # in pixels
midden_coords[1] = (midden_coords[1]-THERMAL_TOP)/THERMAL_PIXEL_HEIGHT - THERMAL_START_ROW # in pixels
midden_coords = np.around(midden_coords).astype(int)
midden_matrix = np.zeros((thermal_orthomosaic.shape[0],thermal_orthomosaic.shape[1])).astype(int)

for loc in midden_coords.T:
    midden_matrix[loc[1],loc[0]] = 1

print(np.sum(midden_matrix)) # 52 middens
np.save('data/midden-matrix', midden_matrix) # save midden locations in orthomosaic as numpy array