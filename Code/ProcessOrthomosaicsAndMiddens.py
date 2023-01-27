''' Process Orthomosaics by Lucia Gordon & Samuel Collier '''
# Imports
from numpy import array, zeros, ma, around, all, sum, amax, amin, save, any, vstack, hstack, ceil, load, float32
from matplotlib.pyplot import figure, imshow, xlabel, ylabel, colorbar, show, savefig, axis, close
from osgeo import gdal
from shutil import rmtree
from os import mkdir, remove, path
from sys import argv

# Paths
thermalTiffPath = 'Tiffs/Firestorm3/Thermal/Merged.tif' # change as needed
RGBTiffPath = 'Tiffs/Firestorm3/RGB/Merged.tif' # change as needed
middenLocationsPath = 'Phase3/MiddenLocations.npy' # change as needed
LiDARTiffPath = 'Tiffs/Firestorm3/LiDAR/DTM25cm.tif'

# Preparing folders
for imageryType in ['Thermal', 'RGB', 'LiDAR']:
    if path.exists('Phase3/Images/'+imageryType):
        rmtree('Phase3/Images/'+imageryType)

    if path.exists('Phase3/Data/'+imageryType):
        rmtree('Phase3/Data/'+imageryType)

    mkdir('Phase3/Images/'+imageryType)
    mkdir('Phase3/Data/'+imageryType)

# Functions
def cropArray(matrix):
    startRow = 0
    endRow = matrix.shape[0]
    startCol = 0
    endCol = matrix.shape[1]

    for rowIndex in range(len(matrix)):
        if any(matrix[rowIndex]!=0):
            startRow = rowIndex
            break

    matrix = matrix[startRow:]

    for rowIndex in range(len(matrix)): 
        if all(matrix[rowIndex]==0):
            endRow = rowIndex
            break
        else:
            endRow = matrix.shape[0]

    matrix = matrix[:endRow]
    
    for colIndex in range(len(matrix.T)):
        if any(matrix.T[colIndex]!=0):
            startCol = colIndex
            break

    matrix = matrix.T[startCol:].T

    for colIndex in range(len(matrix.T)):
        if all(matrix.T[colIndex]==0):
            endCol = colIndex
            break
        else:
            endCol = matrix.shape[1]

    matrix = matrix.T[:endCol].T

    return startRow, startRow+endRow, startCol, startCol+endCol

def plotImage(imageryType, imageData):
    figure(imageryType + 'Orthomosaic', dpi=200)
    image = imshow(imageData)
    
    if imageryType == 'Thermal' or imageryType == 'LiDAR':
        image.set_cmap('inferno')

    axis('off') # remove axes        
    savefig('Phase3/Images/' + imageryType + '/' + imageryType + 'Orthomosaic.png', bbox_inches='tight', pad_inches=0)
    close()

# Process thermal orthomosaic
thermalInterval = 40 # width of cropped thermal images in pixels
thermalStride = 10 # overlap of cropped thermal images in pixels
thermalDataset = gdal.Open(thermalTiffPath) # converts the tiff to a Dataset object
thermalNumRows = thermalDataset.RasterYSize # 4400 pixels
thermalNumCols = thermalDataset.RasterXSize # 4400 pixels
thermalNumBands = thermalDataset.RasterCount # 4 bands
thermalYOrigin = thermalDataset.GetGeoTransform()[3] # 7195100 m
thermalPixelHeight = thermalDataset.GetGeoTransform()[5] # -0.5 m
thermalXOrigin = thermalDataset.GetGeoTransform()[0] # 7195100 m
thermalPixelWidth = thermalDataset.GetGeoTransform()[1] # 0.5 m
thermalTop = thermalYOrigin # 7195100 m
thermalBottom = thermalYOrigin + thermalPixelHeight * thermalNumRows # 7192900 m
thermalLeft = thermalXOrigin # 352000 m
thermalRight = thermalXOrigin + thermalPixelWidth * thermalNumCols # 354200 m
thermalBand = ((thermalDataset.GetRasterBand(4)).ReadAsArray(0,0,thermalNumCols,thermalNumRows).astype(float32)) # 4th band corresponds to thermal data
thermalOrthomosaicMin = amin(ma.masked_less(thermalBand,2000)) # 7638 = min pixel value in orthomosaic, excluding background
thermalOrthomosaic = ma.masked_less(thermalBand-thermalOrthomosaicMin,0).filled(0) # downshift the pixel values such that the min of the orthomosaic is 0 and set the background pixels to 0
print(thermalOrthomosaic.shape)
thermalStartRow, thermalEndRow, thermalStartCol, thermalEndCol = cropArray(thermalOrthomosaic) # extract indices for cropping
thermalOrthomosaic = thermalOrthomosaic[thermalStartRow:thermalEndRow,thermalStartCol:thermalEndCol] # crop out rows and columns that are 0
print(thermalOrthomosaic.shape)
newThermalRows = zeros((int(ceil((thermalOrthomosaic.shape[0]-thermalInterval)/(thermalInterval/2+thermalStride)))*int(thermalInterval/2+thermalStride)+thermalInterval-thermalOrthomosaic.shape[0],thermalOrthomosaic.shape[1])) # add rows so that nothing gets cut off in cropping
thermalOrthomosaic = vstack((thermalOrthomosaic, newThermalRows)) # add rows to bottom of thermal orthomosaic
newThermalColumns = zeros((thermalOrthomosaic.shape[0],int(ceil((thermalOrthomosaic.shape[1]-thermalInterval)/(thermalInterval/2+thermalStride)))*int(thermalInterval/2+thermalStride)+thermalInterval-thermalOrthomosaic.shape[1])) # add columns so that nothing gets cut off in cropping
thermalOrthomosaic = hstack((thermalOrthomosaic, newThermalColumns)) # add columns to right of thermal orthomosaic
print(thermalOrthomosaic.shape)

save('Phase3/Data/Thermal/ThermalOrthomosaicMatrix', thermalOrthomosaic) # save thermal orthomosaic as numpy array
plotImage('Thermal', thermalOrthomosaic) # plot thermal orthomosaic

# Process RGB orthomosaic
RGBInterval = 400 # width of cropped RGB images in pixels
RGBStride = 100 # overlap of cropped RGB images in pixels
RGBDataset = gdal.Open(RGBTiffPath) # converts the tiff to a Dataset object
RGBNumRows = RGBDataset.RasterYSize # 54000 pixels
RGBNumCols = RGBDataset.RasterXSize # 54000 pixels
RGBNumBands = RGBDataset.RasterCount # 3 bands
RGBYOrigin = RGBDataset.GetGeoTransform()[3] # 7195500 m
RGBPixelHeight = RGBDataset.GetGeoTransform()[5] # -0.05 m
RGBXOrigin = RGBDataset.GetGeoTransform()[0] # 7195500 m
RGBPixelWidth = RGBDataset.GetGeoTransform()[1] # 0.05 m
RGBTop = RGBYOrigin # 7195500 m
RGBBottom = RGBYOrigin + RGBPixelHeight * RGBNumRows # 7192800 m
RGBLeft = RGBXOrigin # 351900 m
RGBRight = RGBXOrigin + RGBPixelWidth * RGBNumCols # 354600 m
RGBBands = zeros((RGBNumRows,RGBNumCols,RGBNumBands)) # empty RGB orthomosaic

for band in range(RGBNumBands):
    RGBBands[:,:,band] = (RGBDataset.GetRasterBand(band+1)).ReadAsArray(0,0,RGBNumCols,RGBNumRows) # add band data to RGB orthomosaic

RGBOrthomosaic = RGBBands[int((thermalTop-RGBTop)/RGBPixelHeight):int(RGBBands.shape[0]+(thermalBottom-RGBBottom)/RGBPixelHeight),int((thermalLeft-RGBLeft)/RGBPixelWidth):int(RGBBands.shape[1]+(thermalRight-RGBRight)/RGBPixelWidth)].astype('uint8') # crop the RGB orthomosaic to cover the same area as the thermal orthomosaic
print(RGBOrthomosaic.shape)
RGBOrthomosaic = RGBOrthomosaic[10*thermalStartRow:10*thermalEndRow,10*thermalStartCol:10*thermalEndCol] # crop RGB orthomosaic to cover the same area as the thermal orthomosaic after removing empty rows and columns
print(RGBOrthomosaic.shape)
newRGBRows = zeros((int(ceil((RGBOrthomosaic.shape[0]-RGBInterval)/(RGBInterval/2+RGBStride)))*int(RGBInterval/2+RGBStride)+RGBInterval-RGBOrthomosaic.shape[0],RGBOrthomosaic.shape[1],RGBOrthomosaic.shape[2])) # add rows so that nothing gets cut off in cropping
RGBOrthomosaic = vstack((RGBOrthomosaic, newRGBRows)) # add rows to bottom of RGB orthomosaic
newRGBColumns = zeros((RGBOrthomosaic.shape[0],int(ceil((RGBOrthomosaic.shape[1]-RGBInterval)/(RGBInterval/2+RGBStride)))*int(RGBInterval/2+RGBStride)+RGBInterval-RGBOrthomosaic.shape[1],RGBOrthomosaic.shape[2])) # add columns so that nothing gets cut off in cropping
RGBOrthomosaic = hstack((RGBOrthomosaic, newRGBColumns)).astype('uint8') # add columns to right of RGB orthomosaic
print(RGBOrthomosaic.shape)
save('Phase3/Data/RGB/RGBOrthomosaicMatrix', RGBOrthomosaic) # save RGB orthomosaic as numpy array
plotImage('RGB', RGBOrthomosaic) # plot RGB orthomosaic

# Process LiDAR orthomosaic
LiDARInterval = 80
LiDARStride = 20
LiDARDataset = gdal.Open(LiDARTiffPath)
LiDARNumRows = LiDARDataset.RasterYSize # 10117 pixels
LiDARNumCols = LiDARDataset.RasterXSize # 11769 pixels
LiDARNumBands = LiDARDataset.RasterCount # 1 band
LiDARYOrigin = LiDARDataset.GetGeoTransform()[3] # 7195053.75 m
LiDARPixelHeight = LiDARDataset.GetGeoTransform()[5] # -0.25 m
LiDARXOrigin = LiDARDataset.GetGeoTransform()[0] # 351557.5 m
LiDARPixelWidth = LiDARDataset.GetGeoTransform()[1] # 0.25 m
LiDARTop = LiDARYOrigin # 7195053.75 m
LiDARBottom = LiDARYOrigin + LiDARPixelHeight * LiDARNumRows # 7192524.5 m
LiDARLeft = LiDARXOrigin # 351557.5 m
LiDARRight = LiDARXOrigin + LiDARPixelWidth * LiDARNumCols # 354499.75 m
LiDARBand = (LiDARDataset.GetRasterBand(1)).ReadAsArray(0,0,LiDARNumCols,LiDARNumRows)
LiDAROrthomosaic = ma.masked_equal(LiDARBand,-9999).filled(0)
LiDAROrthomosaic = vstack((zeros((185,LiDAROrthomosaic.shape[1])), LiDAROrthomosaic))
LiDAROrthomosaic = LiDAROrthomosaic[:int(LiDAROrthomosaic.shape[0]+(thermalBottom-LiDARBottom)/LiDARPixelHeight),int((thermalLeft-LiDARLeft)/LiDARPixelWidth):int(LiDAROrthomosaic.shape[1]+(thermalRight-LiDARRight)/LiDARPixelWidth)] # crop the LiDAR orthomosaic to cover the same area as the thermal orthomosaic
print(LiDAROrthomosaic.shape)
LiDAROrthomosaic = LiDAROrthomosaic[2*thermalStartRow:2*thermalEndRow,2*thermalStartCol:2*thermalEndCol] # crop LiDAR orthomosaic to cover the same area as the thermal orthomosaic after removing empty rows and columns
print(LiDAROrthomosaic.shape)
newLiDARRows = zeros((int(ceil((LiDAROrthomosaic.shape[0]-LiDARInterval)/(LiDARInterval/2+LiDARStride)))*int(LiDARInterval/2+LiDARStride)+LiDARInterval-LiDAROrthomosaic.shape[0],LiDAROrthomosaic.shape[1])) # add rows so that nothing gets cut off in cropping
LiDAROrthomosaic = vstack((LiDAROrthomosaic, newLiDARRows)) # add rows to bottom of LiDAR orthomosaic
newLiDARColumns = zeros((LiDAROrthomosaic.shape[0],int(ceil((LiDAROrthomosaic.shape[1]-LiDARInterval)/(LiDARInterval/2+LiDARStride)))*int(LiDARInterval/2+LiDARStride)+LiDARInterval-LiDAROrthomosaic.shape[1])) # add columns so that nothing gets cut off in cropping
LiDAROrthomosaic = hstack((LiDAROrthomosaic, newLiDARColumns)) # add columns to right of LiDAR orthomosaic
print(LiDAROrthomosaic.shape)
save('Phase3/Data/LiDAR/LiDAROrthomosaicMatrix', LiDAROrthomosaic) # save LiDAR orthomosaic as numpy array
plotImage('LiDAR', LiDAROrthomosaic) # plot LiDAR orthomosaic

# Process middens
middenCoords = load(middenLocationsPath).T # in meters
# middenCoords[0] = (middenCoords[0]-thermalXOrigin)/thermalPixelWidth - thermalStartCol # in pixels
# middenCoords[1] = (middenCoords[1]-thermalYOrigin)/thermalPixelHeight - thermalStartRow # in pixels
# middenCoords = around(middenCoords).astype(int)
# middenLocsInOrthomosaic = zeros((thermalOrthomosaic.shape[0],thermalOrthomosaic.shape[1])).astype(int)

# for index in range(len(middenCoords.T)):
#     if middenCoords.T[index][1] >= thermalOrthomosaic.shape[0]:
#         print(index, middenCoords.T[index])
#     if middenCoords.T[index][0] >= thermalOrthomosaic.shape[1]:
#         print(index, middenCoords.T[index])

middenCoords[0] = (middenCoords[0]-thermalXOrigin)/RGBPixelWidth - 10*thermalStartCol # in pixels
middenCoords[1] = (middenCoords[1]-thermalYOrigin)/RGBPixelHeight - 10*thermalStartRow # in pixels - sign
middenCoords = around(middenCoords).astype(int)
middenLocsInOrthomosaic = zeros((RGBOrthomosaic.shape[0], RGBOrthomosaic.shape[1])).astype(int)

for index in range(len(middenCoords.T)):
    if middenCoords.T[index][1] >= RGBOrthomosaic.shape[0]:
        print(index, middenCoords.T[index])
    if middenCoords.T[index][0] >= RGBOrthomosaic.shape[1]:
        print(index, middenCoords.T[index])

for loc in middenCoords.T:
    middenLocsInOrthomosaic[loc[1],loc[0]] = 1

if path.exists('Phase3/Data/MiddenMatrix'):
    remove('Phase3/Data/MiddenMatrix')

save('Phase3/Data/MiddenMatrix', middenLocsInOrthomosaic)