from numpy import load, around, append, zeros, vstack, ceil, hstack, amin, ma, amax
from cv2 import imread
from matplotlib.pyplot import figure, imshow, axis, savefig, close, colorbar
from PIL import Image
import PIL
from osgeo import gdal

# from PIL import Image

# background = Image.open('Phase3/Images/Thermal/ThermalOrthomosaic.png')
# # overlay = Image.open('Phase3/Images/RGB/RGBOrthomosaic.png')
# overlay = Image.open('Phase3/Images/LiDAR/LiDAROrthomosaic.png')

# background = background.convert("RGBA")
# overlay = overlay.convert("RGBA")

# new_img = Image.blend(background, overlay, 0.5)
# new_img.save('OverlainOrthomosaics.png','PNG')

# middenLocationsPath = 'Phase3/MiddenLocations.npy' # change as needed
# middenCoords = load(middenLocationsPath).T # in meters
# # middenCoords[0] = (middenCoords[0]-thermalXOrigin)/thermalPixelWidth - thermalStartCol # in pixels
# # middenCoords[1] = (middenCoords[1]-thermalYOrigin)/thermalPixelHeight - thermalStartRow # in pixels
# middenCoords = around(middenCoords).astype(int)
# # middenLocsInOrthomosaic = zeros((thermalOrthomosaic.shape[0],thermalOrthomosaic.shape[1])).astype(int)
# print(len(middenCoords[0]))
# for index in range(len(middenCoords.T)):
#     if middenCoords.T[index][1] >= thermalOrthomosaic.shape[0]:
#         print(index, middenCoords.T[index])
#     if middenCoords.T[index][0] >= thermalOrthomosaic.shape[1]:
#         print(index, middenCoords.T[index])

# for loc in middenCoords.T:
#     middenLocsInOrthomosaic[loc[1],loc[0]] = 1

# print(sum(middenLocsInOrthomosaic))
# if path.exists('Phase3/Data/MiddenMatrix'):
#     remove('Phase3/Data/MiddenMatrix')

# save('Phase3/Data/MiddenMatrix', middenLocsInOrthomosaic)

# arr = imread('Phase3/Images/Thermal/Middens/Midden0.png')

# figure(dpi=60.7) # to get resultant arrays of (224,224,3)
# imshow(arr) # plot the array of pixel values as an image
# axis('off') # remove axes        
# savefig('Phase3/TestMidden0.png', bbox_inches='tight', pad_inches=0) # save the image containing a midden
# close() # close the image to save memory

# Rotate png
# img = Image.open('Phase3/Images/Thermal/Middens/Midden0.png').rotate(180)
# img.save('Phase3/TestRotMidden0.png')

# View LiDAR tiff
LiDARTiffPath = 'Tiffs/Firestorm3/LiDAR/DTM25cm.tif'
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
LiDARBand = (LiDARDataset.GetRasterBand(1)).ReadAsArray(0,0,LiDARNumCols,LiDARNumRows) # 4th band corresponds to thermal data

figure('LiDAR Orthomosaic OG', dpi=200)
image = imshow(LiDARBand)
image.set_cmap('inferno')
axis('off') # remove axes        
savefig('Phase3/LiDAROrthomosaicOG.png', bbox_inches='tight', pad_inches=0)
close()


# LiDAROrthomosaicMin = amin(ma.masked_less(LiDARBand,2000)) # 7638 = min pixel value in orthomosaic, excluding background
LiDAROrthomosaic = ma.masked_equal(LiDARBand,-9999).filled(0) # downshift the pixel values such that the min of the orthomosaic is 0 and set the background pixels to 0

print(amin(LiDAROrthomosaic))
print(amax(LiDAROrthomosaic))

thermalTop = 7195100
thermalBottom = 7192900
thermalLeft = 352000
thermalRight = 354200

thermalStartRow = 495
thermalEndRow = 4400
thermalStartCol = 339
thermalEndCol = 3716

LiDAROrthomosaic = vstack((zeros((185,LiDAROrthomosaic.shape[1])), LiDAROrthomosaic))
LiDAROrthomosaic = LiDAROrthomosaic[:int(LiDAROrthomosaic.shape[0]+(thermalBottom-LiDARBottom)/LiDARPixelHeight),int((thermalLeft-LiDARLeft)/LiDARPixelWidth):int(LiDAROrthomosaic.shape[1]+(thermalRight-LiDARRight)/LiDARPixelWidth)] # crop the LiDAR orthomosaic to cover the same area as the thermal orthomosaic
LiDAROrthomosaic = LiDAROrthomosaic[2*thermalStartRow:2*thermalEndRow,2*thermalStartCol:2*thermalEndCol] # crop RGB orthomosaic to cover the same area as the thermal orthomosaic after removing empty rows and columns
newLiDARRows = zeros((int(ceil((LiDAROrthomosaic.shape[0]-LiDARInterval)/(LiDARInterval/2+LiDARStride)))*int(LiDARInterval/2+LiDARStride)+LiDARInterval-LiDAROrthomosaic.shape[0],LiDAROrthomosaic.shape[1])) # add rows so that nothing gets cut off in cropping
LiDAROrthomosaic = vstack((LiDAROrthomosaic, newLiDARRows)) # add rows to bottom of RGB orthomosaic
newLiDARColumns = zeros((LiDAROrthomosaic.shape[0],int(ceil((LiDAROrthomosaic.shape[1]-LiDARInterval)/(LiDARInterval/2+LiDARStride)))*int(LiDARInterval/2+LiDARStride)+LiDARInterval-LiDAROrthomosaic.shape[1])) # add columns so that nothing gets cut off in cropping
LiDAROrthomosaic = hstack((LiDAROrthomosaic, newLiDARColumns)) # add columns to right of RGB orthomosaic
# save('Phase3/Data/LiDAR/LiDAROrthomosaicMatrix', RGBOrthomosaic) # save RGB orthomosaic as numpy array
print(LiDAROrthomosaic.shape)
figure('LiDAR Orthomosaic', dpi=200)
image = imshow(LiDAROrthomosaic)
image.set_cmap('inferno')
colorbar()
axis('off') # remove axes        
savefig('Phase3/LiDAROrthomosaic.png', bbox_inches='tight', pad_inches=0)
close()


print(LiDARNumRows)
print(LiDARNumCols)
print(LiDARNumBands)
print(LiDARYOrigin)
print(LiDARPixelHeight)
print(LiDARXOrigin)
print(LiDARPixelWidth)
print(LiDARTop)
print(LiDARBottom)
print(LiDARLeft)
print(LiDARRight)