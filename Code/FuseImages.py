'''Fuse Images by Lucia Gordon'''
# Imports
from PIL import Image
from os import mkdir, path, remove, listdir
from shutil import rmtree
from cv2 import imread
from numpy import array, save

# Preparing folders
if path.exists('Phase3/Images/Fused'):
    rmtree('Phase3/Images/Fused')

if path.exists('Phase3/Data/Fused'):
    rmtree('Phase3/Data/Fused')

mkdir('Phase3/Images/Fused')
mkdir('Phase3/Images/Fused/Middens')
mkdir('Phase3/Images/Fused/Rotated90Middens')
mkdir('Phase3/Images/Fused/Rotated180Middens')
mkdir('Phase3/Images/Fused/Rotated270Middens')
mkdir('Phase3/Images/Fused/Empty')   
mkdir('Phase3/Data/Fused')
mkdir('Phase3/Data/Fused/Middens')     
mkdir('Phase3/Data/Fused/Rotated90Middens')
mkdir('Phase3/Data/Fused/Rotated180Middens')
mkdir('Phase3/Data/Fused/Rotated270Middens')
mkdir('Phase3/Data/Fused/Empty')

# Fusing orthomosaics
thermalOrthomosaic = (Image.open('Phase3/Images/Thermal/ThermalOrthomosaic.png')).convert('RGBA')
RGBOrthomosaic = (Image.open('Phase3/Images/RGB/RGBOrthomosaic.png')).convert('RGBA')
fusedOrthomosaic = Image.blend(thermalOrthomosaic, RGBOrthomosaic, 0.5)
fusedOrthomosaic.save('Phase3/Images/Fused/FusedOrthomosaic.png','PNG')

# Fuse images as PNGs
rotated90MiddenPNGArrays = []
rotated180MiddenPNGArrays = []
rotated270MiddenPNGArrays = []

for index in range(len(listdir('Phase3/Images/Thermal/Middens'))):
    thermalMidden = Image.open('Phase3/Images/Thermal/Middens/Midden'+str(index)+'.png')
    RGBMidden = Image.open('Phase3/Images/RGB/Middens/Midden'+str(index)+'.png')
    fusedMidden = Image.blend(thermalMidden, RGBMidden, 0.5)
    fusedMidden.save('Phase3/Images/Fused/Middens/Midden'+str(index)+'.png','PNG')
    rotated90 = Image.open('Phase3/Images/Fused/Middens/Midden'+str(index)+'.png').rotate(90)
    rotated90.save('Phase3/Images/Fused/Rotated90Middens/Rotated90Midden'+str(index)+'.png')
    rotated90MiddenPNGArrays.append(imread('Phase3/Images/Fused/Rotated90Middens/Rotated90Midden'+str(index)+'.png')) # convert each midden image PNG to an RGB array
    rotated180 = Image.open('Phase3/Images/Fused/Middens/Midden'+str(index)+'.png').rotate(180)
    rotated180.save('Phase3/Images/Fused/Rotated180Middens/Rotated180Midden'+str(index)+'.png')
    rotated180MiddenPNGArrays.append(imread('Phase3/Images/Fused/Rotated180Middens/Rotated180Midden'+str(index)+'.png')) # convert each midden image PNG to an RGB array
    rotated270 = Image.open('Phase3/Images/Fused/Middens/Midden'+str(index)+'.png').rotate(270)
    rotated270.save('Phase3/Images/Fused/Rotated270Middens/Rotated270Midden'+str(index)+'.png')
    rotated270MiddenPNGArrays.append(imread('Phase3/Images/Fused/Rotated270Middens/Rotated270Midden'+str(index)+'.png')) # convert each midden image PNG to an RGB array

for index in range(len(listdir('Phase3/Images/Thermal/Empty'))):
    thermalEmpty = Image.open('Phase3/Images/Thermal/Empty/Empty'+str(index)+'.png')
    RGBEmpty = Image.open('Phase3/Images/RGB/Empty/Empty'+str(index)+'.png')
    fusedEmpty = Image.blend(thermalEmpty, RGBEmpty, 0.5)
    fusedEmpty.save('Phase3/Images/Fused/Empty/Empty'+str(index)+'.png','PNG')

# Convert fused PNGs to arrays
middenPNGArrays = []
emptyPNGArrays = []

for index in range(len(listdir('Phase3/Images/Fused/Middens'))):
    middenPNGArrays.append(imread('Phase3/Images/Fused/Middens/Midden'+ str(index)+'.png')) # convert each midden image PNG to an RGB array

for index in range(len(listdir('Phase3/Images/Fused/Empty'))):
    emptyPNGArrays.append(imread('Phase3/Images/Fused/Empty/Empty'+str(index)+'.png')) # convert each empty image PNG to an RGB array

print(array(middenPNGArrays).shape)
print(array(emptyPNGArrays).shape)

save('Phase3/Data/Fused/MiddenImages', middenPNGArrays) # save the midden PNG arrays
save('Phase3/Data/Fused/Rotated90MiddenImages', rotated90MiddenPNGArrays) # save the midden PNG arrays
save('Phase3/Data/Fused/Rotated180MiddenImages', rotated180MiddenPNGArrays) # save the midden PNG arrays
save('Phase3/Data/Fused/Rotated270MiddenImages', rotated270MiddenPNGArrays) # save the midden PNG arrays
save('Phase3/Data/Fused/EmptyImages', emptyPNGArrays) # save the empty PNG arrays