# imports
import csv
import os
import random
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import utils
from sys import argv
from shapely.geometry import Polygon

# global variables
project_dir = utils.get_project_dir()
site = utils.get_site()
identifiers = np.load(f'{project_dir}/{site}/data/identifiers.npy')
print(f'identifiers length = {len(identifiers)}')
labeled_indices = np.load(f'{project_dir}/{site}/data/labeled-indices.npy')
labeled_ids_labels = np.load(f'{project_dir}/{site}/data/labeled-ids.npy').T[1]
labels = np.load(f'{project_dir}/{site}/data/labels.npy')
constants = np.load(f'{project_dir}/{site}/data/constants.npy')
THERMAL_INTERVAL = int(constants[0][1])
THERMAL_STRIDE = int(constants[1][1])
THERMAL_LEFT = float(constants[2][1])
THERMAL_TOP = float(constants[3][1])
THERMAL_PIXEL_WIDTH = float(constants[4][1])
THERMAL_PIXEL_HEIGHT = float(constants[5][1])
NUM_HORIZONTAL = int(constants[8][1])
THRESHOLD = 0.7

# functions
def get_image_center_pixels(identifier):
    row = np.floor(identifier / NUM_HORIZONTAL)
    col = identifier - NUM_HORIZONTAL * np.floor(identifier / NUM_HORIZONTAL)
    x_pixels = col * (THERMAL_STRIDE + THERMAL_INTERVAL / 2) + THERMAL_INTERVAL / 2
    y_pixels = row * (THERMAL_STRIDE + THERMAL_INTERVAL / 2) + THERMAL_INTERVAL / 2

    return x_pixels, y_pixels

def get_image_center_meters(x_pixels, y_pixels):
    x = THERMAL_LEFT + x_pixels * THERMAL_PIXEL_WIDTH
    y = THERMAL_TOP + y_pixels * THERMAL_PIXEL_HEIGHT

    return x, y

scores = np.load(f'{project_dir}/{site}/data/scores.npy')
scores = np.array([labels[i] if i in labeled_indices else scores[i] for i in range(len(scores))]) # ensuring the scores of labeled images match the label
print(f'Scores length = {len(scores)}')
likely_midden_indices = np.where(scores >= THRESHOLD)[0]
print(f'Likely midden indices length = {len(likely_midden_indices)}')
likely_midden_identifiers = identifiers[likely_midden_indices]
print(f'Likely midden identifier length = {len(likely_midden_identifiers)}')

def make_shapefile(identifiers):
    centers_in_meters = []

    for identifier in identifiers:
        x_pixels, y_pixels = get_image_center_pixels(identifier)
        x, y = get_image_center_meters(x_pixels, y_pixels)
        centers_in_meters.append([x, y])

    polygons = [Polygon([[center[0] - 10, center[1] - 10], [center[0] + 10, center[1] - 10], [center[0] + 10, center[1] + 10], [center[0] - 10, center[1] + 10]]) for center in centers_in_meters] # each image is 20 x 20 m
    gdf = gpd.GeoDataFrame(geometry = polygons)
    gdf['id'] = identifiers
    gdf['label'] = [labels[likely_midden_indices[i]] if likely_midden_indices[i] in labeled_indices else '' for i in range(len(identifiers))]
    if os.path.exists(f'{project_dir}/{site}/data/shapefile/'): shutil.rmtree(f'{project_dir}/{site}/data/shapefile/')
    os.mkdir(f'{project_dir}/{site}/data/shapefile/')
    gdf.to_file(f'{project_dir}/{site}/data/shapefile/{site}-{int(100*THRESHOLD)}-percent-middens-shapefile.shp')
    shutil.make_archive(f'{project_dir}/{site}/data/{site}-{int(100*THRESHOLD)}-percent-middens-shapefile', 'zip', f'{project_dir}/{site}/data/shapefile')

make_shapefile(likely_midden_identifiers)

all_labels = np.load(f'{project_dir}/{site}/data/all-labels/labels.npy')
print(np.sum(all_labels))
midden_indices = np.where(all_labels == 1)[0]

missing_middens = [index for index in midden_indices if index not in likely_midden_indices]
print(len(missing_middens))
print(np.sum(labeled_ids_labels))
#
# centers_in_meters = []

# for identifier in likely_midden_identifiers:
#     x_pixels, y_pixels = get_image_center_pixels(identifier)
#     x, y = get_image_center_meters(x_pixels, y_pixels)
#     centers_in_meters.append([x,y])

# centers_in_meters = np.array(centers_in_meters)
# print(centers_in_meters.shape)

# polygons = [Polygon([[center[0] - 10, center[1] - 10], [center[0] + 10, center[1] - 10], [center[0] + 10, center[1] + 10], [center[0] - 10,center[1] + 10]]) for center in centers_in_meters]
# gdf = geopandas.GeoDataFrame(geometry = polygons)
# gdf.to_file('firestorm-1/data/shapefile/75-percent-midden-boxes.shp')
# shutil.make_archive('firestorm-1/data/shapefile', 'zip', 'firestorm-1/data/shapefile')

# with open('firestorm-1/data/75-percent-midden-centers-m.csv', 'w') as f:
#     write = csv.writer(f)
#     write.writerow(['x','y'])
#     write.writerows(centers_in_meters)