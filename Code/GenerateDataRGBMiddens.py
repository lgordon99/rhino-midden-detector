'''Generate Data by Lucia Gordon'''
# Imports
from numpy import load, amax, amin, flip, arange, save, array, all, sum, take, sort, flip
from shutil import rmtree
from os import mkdir, path, remove
from random import sample
from matplotlib.pyplot import figure, imshow, axis, savefig, close
from cv2 import imread
from PIL import Image
import PIL

# Paths
thermalOrthomosaicPath = 'Phase3/Data/Thermal/ThermalOrthomosaicMatrix.npy'
RGBOrthomosaicPath = 'Phase3/Data/RGB/RGBOrthomosaicMatrix.npy'
LiDAROrthomosaicPath = 'Phase3/Data/LiDAR/LiDAROrthomosaicMatrix.npy'
middenMatrixPath = 'Phase3/Data/MiddenMatrix.npy'

# Preparing folders
for imageryType in ['Thermal', 'RGB', 'LiDAR']:
    if path.exists('Phase3/Images/'+imageryType+'/Middens'):
        rmtree('Phase3/Images/'+imageryType+'/Middens')

    if path.exists('Phase3/Images/'+imageryType+'/Rotated90Middens'):
        rmtree('Phase3/Images/'+imageryType+'/Rotated90Middens')
    
    if path.exists('Phase3/Images/'+imageryType+'/Rotated180Middens'):
        rmtree('Phase3/Images/'+imageryType+'/Rotated180Middens')
    
    if path.exists('Phase3/Images/'+imageryType+'/Rotated270Middens'):
        rmtree('Phase3/Images/'+imageryType+'/Rotated270Middens')

    if path.exists('Phase3/Images/'+imageryType+'/Empty'):
        rmtree('Phase3/Images/'+imageryType+'/Empty')

    mkdir('Phase3/Images/'+imageryType+'/Middens')  
    mkdir('Phase3/Images/'+imageryType+'/Rotated90Middens')   
    mkdir('Phase3/Images/'+imageryType+'/Rotated180Middens')   
    mkdir('Phase3/Images/'+imageryType+'/Rotated270Middens')   
    mkdir('Phase3/Images/'+imageryType+'/Empty')

# Functions
def matrixToPNG(imageryType, matrix, path):
    figure(dpi=60.7) # dpi=60.7 to get resultant arrays of (224,224,3), dpi=11 to get resultant arrays of (40,40,3)
    image = imshow(matrix) # plot the array of pixel values as an image

    if imageryType == 'Thermal' or imageryType == 'LiDAR':
        image.set_cmap('inferno')
    
    axis('off') # remove axes        
    savefig('Phase3/Images/'+imageryType+path+'.png', bbox_inches='tight', pad_inches=0) # save the image containing a midden
    close() # close the image to save memory

def saveData(imageryType, middenImages, emptyImages):
    middenPNGArrays = []
    rotated90MiddenPNGArrays = []
    rotated180MiddenPNGArrays = []
    rotated270MiddenPNGArrays = []
    emptyPNGArrays = []

    for index in range(len(middenImages)):
        matrixToPNG(imageryType, middenImages[index], '/Middens/Midden'+str(index)) # convert the matrix corresponding to a midden image to a PNG
    
    for index in range(len(emptyImages)):
        matrixToPNG(imageryType, emptyImages[index], '/Empty/Empty'+str(index)) # convert the matrix corresponding to an empty image to a PNG

    for index in range(len(middenImages)):
        middenPNGArrays.append(imread('Phase3/Images/'+imageryType+'/Middens/Midden'+str(index)+'.png')) # convert each midden image PNG to an RGB array
        rotated90 = Image.open('Phase3/Images/'+imageryType+'/Middens/Midden'+str(index)+'.png').rotate(90)
        rotated90.save('Phase3/Images/'+imageryType+'/Rotated90Middens/Rotated90Midden'+str(index)+'.png')
        rotated90MiddenPNGArrays.append(imread('Phase3/Images/'+imageryType+'/Rotated90Middens/Rotated90Midden'+ str(index)+'.png')) # convert each midden image PNG to an RGB array
        rotated180 = Image.open('Phase3/Images/'+imageryType+'/Middens/Midden'+str(index)+'.png').rotate(180)
        rotated180.save('Phase3/Images/'+imageryType+'/Rotated180Middens/Rotated180Midden'+str(index)+'.png')
        rotated180MiddenPNGArrays.append(imread('Phase3/Images/'+imageryType+'/Rotated180Middens/Rotated180Midden'+ str(index)+'.png')) # convert each midden image PNG to an RGB array
        rotated270 = Image.open('Phase3/Images/'+imageryType+'/Middens/Midden'+str(index)+'.png').rotate(270)
        rotated270.save('Phase3/Images/'+imageryType+'/Rotated270Middens/Rotated270Midden'+str(index)+'.png')
        rotated270MiddenPNGArrays.append(imread('Phase3/Images/'+imageryType+'/Rotated270Middens/Rotated270Midden'+ str(index)+'.png')) # convert each midden image PNG to an RGB array

    for index in range(len(emptyImages)):
        emptyPNGArrays.append(imread('Phase3/Images/'+imageryType+'/Empty/Empty'+str(index)+'.png')) # convert each empty image PNG to an RGB array

    save('Phase3/Data/'+imageryType+'/MiddenImages', middenPNGArrays) # save the midden PNG arrays
    save('Phase3/Data/'+imageryType+'/EmptyImages', emptyPNGArrays) # save the empty PNG arrays
    save('Phase3/Data/'+imageryType+'/Rotated90MiddenImages', rotated90MiddenPNGArrays) # save the midden PNG arrays
    save('Phase3/Data/'+imageryType+'/Rotated180MiddenImages', rotated180MiddenPNGArrays) # save the midden PNG arrays
    save('Phase3/Data/'+imageryType+'/Rotated270MiddenImages', rotated270MiddenPNGArrays) # save the midden PNG arrays

# Crop RGB orthomosaic
RGBOrthomosaicMatrix = load(RGBOrthomosaicPath)
middenMatrix = load(middenMatrixPath)
RGBInterval = 400 # width of cropped thermal images in pixels
RGBStride = 100 # overlap of cropped thermal images in pixels
RGBImages = [] # images cropped from orthomosaic
RGBLabelMatrices = [] # midden locations cropped from orthomosaic
middenIndices = [] # list of indices corresponding to cropped images with middens
emptyIndices = [] # list of indices corresponding to cropped images without middens
RGBMiddenImages = [] # subset of the cropped images that contain middens
RGBEmptyImages = [] # subset of the cropped images that are empty
top = 0 # begin cropping from the top of the orthomosaic
bottom = RGBStride + RGBInterval/2 + RGBStride # set the height of the image

while bottom < int(RGBOrthomosaicMatrix.shape[0]): # keep incrementing the bottom value until you hit the bottom of the orthomosaic
    left = 0 # begin cropping from the left end of the orthomosaic
    right = RGBStride + RGBInterval/2 + RGBStride # set the width of the image

    while right < int(RGBOrthomosaicMatrix.shape[1]): # keep incrementing the right value until you hit the right end of the orthomosaic
        croppedImage = RGBOrthomosaicMatrix[int(top):int(bottom),int(left):int(right)].copy() # create an image cropped from the orthomosaic
        RGBImages.append(croppedImage) # save cropped image to list
        RGBLabelMatrices.append(middenMatrix[int(top):int(bottom),int(left):int(right)]) # save the same cropping from the matrix of midden locations
        left += RGBStride + RGBInterval/2 # increment the leftmost boundary for the next image
        right += RGBInterval/2 + RGBStride # increment the rightmost boundary for the next image

    top += RGBStride + RGBInterval/2 # increment the top boundary for the next set of images
    bottom += RGBInterval/2 + RGBStride # increment the bottom boundary for the next set of images

labels = sum(sum(RGBLabelMatrices,axis=1),axis=1) # collapses each label matrix to the number of middens in the corresponding cropped image

for index in range(len(labels)):
    if labels[index] > 1: # if there happens to be more than 1 midden in an image
        labels[index] = 1 # set the label to 1 since we only care about whether there is a midden or not

for index in range(len(labels)):
    if labels[index] == 1: # if the image at the index contains a midden
        middenIndices.append(index)
        RGBMiddenImages.append(RGBImages[index]) # add the image to the list of midden images
    elif labels[index] == 0: # if the image at the index does not contain a midden
        emptyIndices.append(index)
        RGBEmptyImages.append(RGBImages[index]) # add the image to the list of empty images

print(len(RGBMiddenImages))
print(len(RGBEmptyImages))

# Crop thermal orthomosaic
thermalOrthomosaicMatrix = load(thermalOrthomosaicPath)
thermalInterval = 40 # width of cropped thermal images in pixels
thermalStride = 10 # overlap of cropped thermal images in pixels
thermalImages = [] # images cropped from orthomosaic
thermalMiddenImages = [] # subset of the cropped images that contain middens
thermalEmptyImages = [] # subset of the cropped images that are empty
top = 0 # begin cropping from the top of the orthomosaic
bottom = thermalStride + thermalInterval/2 + thermalStride # set the height of the image

while bottom < int(thermalOrthomosaicMatrix.shape[0]): # keep incrementing the bottom value until you hit the bottom of the orthomosaic
    left = 0 # begin cropping from the left end of the orthomosaic
    right = thermalStride + thermalInterval/2 + thermalStride # set the width of the image

    while right < int(thermalOrthomosaicMatrix.shape[1]): # keep incrementing the right value until you hit the right end of the orthomosaic
        croppedImage = thermalOrthomosaicMatrix[int(top):int(bottom),int(left):int(right)].copy() # create an image cropped from the orthomosaic
        croppedImage -= amin(croppedImage) # set the minimum pixel value to 0
        thermalImages.append(croppedImage) # save cropped image to list
        left += thermalStride + thermalInterval/2 # increment the leftmost boundary for the next image
        right += thermalInterval/2 + thermalStride # increment the rightmost boundary for the next image

    top += thermalStride + thermalInterval/2 # increment the top boundary for the next set of images
    bottom += thermalInterval/2 + thermalStride # increment the bottom boundary for the next set of images

for index in range(len(labels)):
    if labels[index] == 1: # if the image at the index contains a midden
        thermalMiddenImages.append(thermalImages[index]) # add the image to the list of midden images
    elif labels[index] == 0: # if the image at the index does not contain a midden
        thermalEmptyImages.append(thermalImages[index]) # add the image to the list of empty images

maxMiddenPixelVals = [amax(thermalMiddenImages[i]) for i in range(len(thermalMiddenImages))]
maxEmptyPixelVals = [amax(thermalEmptyImages[i]) for i in range(len(thermalEmptyImages))]
print(len(maxMiddenPixelVals))
print(len(maxEmptyPixelVals))
print(len(thermalMiddenImages))
print(len(thermalEmptyImages))

# Crop LiDAR orthomosaic
LiDAROrthomosaicMatrix = load(LiDAROrthomosaicPath)
LiDARInterval = 80
LiDARStride = 20
LiDARImages = []
LiDARMiddenImages = []
LiDAREmptyImages = []
top = 0 # begin cropping from the top of the orthomosaic
bottom = LiDARStride + LiDARInterval/2 + LiDARStride # set the height of the image

while bottom < int(LiDAROrthomosaicMatrix.shape[0]): # keep incrementing the bottom value until you hit the bottom of the orthomosaic
    left = 0 # begin cropping from the left end of the orthomosaic
    right = LiDARStride + LiDARInterval/2 + LiDARStride # set the width of the image

    while right < int(LiDAROrthomosaicMatrix.shape[1]): # keep incrementing the right value until you hit the right end of the orthomosaic
        croppedImage = LiDAROrthomosaicMatrix[int(top):int(bottom),int(left):int(right)].copy() # create an image cropped from the orthomosaic
        LiDARImages.append(croppedImage) # save cropped image to list
        left += LiDARStride + LiDARInterval/2 # increment the leftmost boundary for the next image
        right += LiDARInterval/2 + LiDARStride # increment the rightmost boundary for the next image

    top += LiDARStride + LiDARInterval/2 # increment the top boundary for the next set of images
    bottom += LiDARInterval/2 + LiDARStride # increment the bottom boundary for the next set of images

for index in range(len(labels)):
    if labels[index] == 1: # if the image at the index contains a midden
        LiDARMiddenImages.append(LiDARImages[index]) # add the image to the list of midden images
    elif labels[index] == 0: # if the image at the index does not contain a midden
        LiDAREmptyImages.append(LiDARImages[index]) # add the image to the list of empty images

for i in flip(range(len(thermalEmptyImages))):
    if all(thermalEmptyImages[i]==0) and all(RGBEmptyImages[i]==0):
        del thermalEmptyImages[i]
        del RGBEmptyImages[i]
        del LiDAREmptyImages[i]
        del maxEmptyPixelVals[i]

print(len(thermalMiddenImages))
print(len(thermalEmptyImages))
print(len(RGBMiddenImages))
print(len(RGBEmptyImages))
print(len(LiDARMiddenImages))
print(len(LiDAREmptyImages))

save('Phase3/Data/Thermal/RawThermalMiddenImages', thermalMiddenImages)
save('Phase3/Data/Thermal/RawThermalEmptyImages', thermalEmptyImages)
save('Phase3/Data/LiDAR/RawLiDARMiddenImages', LiDARMiddenImages)
save('Phase3/Data/LiDAR/RawLiDAREmptyImages', LiDAREmptyImages)
save('Phase3/Data/Thermal/MaxMiddenPixelVals', maxMiddenPixelVals) # save max pixel values as numpy array
save('Phase3/Data/Thermal/MaxEmptyPixelVals', maxEmptyPixelVals) # save max pixel values as numpy array
saveData('Thermal', thermalMiddenImages, thermalEmptyImages) # save the thermal images as PNGs
saveData('RGB', RGBMiddenImages, RGBEmptyImages)
saveData('LiDAR', LiDARMiddenImages, LiDAREmptyImages)
save('Phase3/Data/Labels', [1] * len(thermalMiddenImages) + [0] * len(thermalEmptyImages)) # save the labels for the PNG arrays