'''process-middens by Lucia Gordon'''

# imports
import pandas as pd
import numpy as np
from sys import argv

folder = argv[1]

# paths
midden_coordinates_path = folder + '/data/midden-coordinates-m.csv'
constants_path = folder + '/data/constants.npy'

# variables
constants = np.load(constants_path)
THERMAL_LEFT_FINAL = float(constants[2][1])
THERMAL_TOP_FINAL = float(constants[3][1])
THERMAL_PIXEL_WIDTH = float(constants[4][1])
THERMAL_PIXEL_HEIGHT = float(constants[5][1])
THERMAL_ORTHOMOSAIC_ROWS = int(constants[6][1])
THERMAL_ORTHOMOSAIC_COLS = int(constants[7][1])

midden_coords = pd.read_csv(midden_coordinates_path).to_numpy().T # in meters
midden_coords[0] = (midden_coords[0] - THERMAL_LEFT_FINAL) / THERMAL_PIXEL_WIDTH # pixels
midden_coords[1] = (midden_coords[1] - THERMAL_TOP_FINAL) / THERMAL_PIXEL_HEIGHT # pixels
midden_coords = np.around(midden_coords).astype(int)

midden_matrix = np.zeros((THERMAL_ORTHOMOSAIC_ROWS, THERMAL_ORTHOMOSAIC_COLS)).astype(int)

for loc in midden_coords.T:
    midden_matrix[loc[1], loc[0]] = 1

print(np.sum(midden_matrix), 'middens') # 52 middens for firestorm 3

np.save(folder + '/data/midden-matrix', midden_matrix) # save midden locations in orthomosaic as numpy array