''' Data Engineering by Lucia Gordon & Samuel Collier '''
from numpy import array, zeros, ma, around, flip, arange, all, sum, amax, amin, uint8, save, where, any, vstack, hstack, ceil, load, mean, std, append
from matplotlib.pyplot import figure, imshow, xlabel, ylabel, colorbar, show, savefig
from osgeo import gdal
from random import sample
from sys import argv
from scipy import ndimage

class DataEngineering:
    def __init__(self, orthomosaicPath, middenLocationsPath, imageryType, part=0):
        self.imageryType = imageryType # Thermal or RGB
        self.part = part # 0 for Thermal, 1 or 2 for RGB

        if self.imageryType == 'Thermal' or self.imageryType+str(self.part) == 'RGB1':
            self.dataset = gdal.Open(orthomosaicPath)
            self.orthomosaic = []

        elif self.imageryType+str(self.part) == 'RGB2':
            self.orthomosaic = load('Data/RGBOrthomosaic.npy')

        self.startCol = 0
        self.endCol = 0
        self.startRow = 0
        self.endRow = 0

        if self.imageryType == 'Thermal':
            self.interval = 40 # 20m / 0.5m/pixel = 40 pixels
            self.stride = 10 # 5m / 0.5m/pixel = 10 pixels

        elif self.imageryType == 'RGB':
            self.interval = 400
            self.stride = 100
            
        self.middenCoords = []

        if self.imageryType == 'Thermal' or self.imageryType+str(self.part) == 'RGB1':
            self.middenLocsInOrthomosaic = []
        
        elif self.imageryType+str(self.part) == 'RGB2':
            self.middenLocsInOrthomosaic = load('Data/RGBMiddenLocsInOrthomosaic.npy')

        self.trainingImages = [] # list of 40x40 pixel images cropped from orthomosaic
        self.trainingLabelMatrices = [] # list of (interval,interval) arrays cropped from the full array of midden locations in the orthomosaic
        self.trainingLabels = []
        self.rotatedImages = []
        self.trainingData = []

        if self.imageryType == 'Thermal' or self.imageryType+str(self.part) == 'RGB1':
            self.getPixelValuesFromTiff()

        self.plotImage(self.orthomosaic, self.imageryType + ' Orthomosaic')

        if self.imageryType == 'Thermal' or self.imageryType+str(self.part) == 'RGB1':
            self.getMiddenLocs(middenLocationsPath)
        
        if self.imageryType == 'Thermal' or self.imageryType+str(self.part) == 'RGB2':
            self.generateTrainingData()
            self.showMiddensOnImage()

    def cropArray(self, matrix):
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

        return matrix, startRow, startRow+endRow, startCol, startCol+endCol

    def getPixelValuesFromTiff(self):
        numCols = self.dataset.RasterXSize
        numRows = self.dataset.RasterYSize
        numBands = self.dataset.RasterCount
        imageData = zeros((numRows, numCols, numBands))

        for band in range(numBands):
            data = self.dataset.GetRasterBand(band+1)
            imageData[:,:,band] = data.ReadAsArray(0,0,numCols,numRows) # (x offset, y offset, x size, y size)

        if self.imageryType == 'Thermal':
            self.orthomosaic = imageData[:,:,3] # extracting fourth band, which corresponds to temperature
            orthomosaicMin = amin(ma.masked_less(self.orthomosaic,2000)) # 7638 = min pixel value in orthomosaic
            self.orthomosaic = ma.masked_less(self.orthomosaic-orthomosaicMin,0).filled(0) # shift the pixel values such that the min of the orthomosaic is 0 and set the background pixels to 0
            self.orthomosaic, self.startRow, _, self.startCol, _ = self.cropArray(self.orthomosaic) # crop out rows and columns that are 0
            newRows = zeros((int(ceil((self.orthomosaic.shape[0]-self.interval)/(self.interval/2+self.stride)))*int(self.interval/2+self.stride)+self.interval-self.orthomosaic.shape[0],self.orthomosaic.shape[1]))
            self.orthomosaic = vstack((self.orthomosaic, newRows))
            newColumns = zeros((self.orthomosaic.shape[0],int(ceil((self.orthomosaic.shape[1]-self.interval)/(self.interval/2+self.stride)))*int(self.interval/2+self.stride)+self.interval-self.orthomosaic.shape[1]))
            self.orthomosaic = hstack((self.orthomosaic, newColumns))
        
        elif self.imageryType == 'RGB':
            self.orthomosaic = imageData.astype('uint8')
            indices = []

            for band in range(self.orthomosaic.shape[2]):
                _, startRow, endRow, startCol, endCol = self.cropArray(self.orthomosaic[:,:,band])
                indices.append([startRow, endRow, startCol, endCol])

            self.startRow = amin(array(indices).T[0], axis=0)
            self.endRow = amax(array(indices).T[1], axis=0)
            self.startCol = amin(array(indices).T[2], axis=0)
            self.endCol = amax(array(indices).T[3], axis=0)
            croppedOrthomosaic = zeros((self.endRow-self.startRow, self.endCol-self.startCol, self.orthomosaic.shape[2]))

            for band in range(self.orthomosaic.shape[2]):
                croppedOrthomosaic[:,:,band] = self.orthomosaic[:,:,band][self.startRow:self.endRow, self.startCol:self.endCol]

            self.orthomosaic = croppedOrthomosaic # crop out rows and columns that are 0
            newRows = zeros((int(ceil((self.orthomosaic.shape[0]-self.interval)/(self.interval/2+self.stride)))*int(self.interval/2+self.stride)+self.interval-self.orthomosaic.shape[0],self.orthomosaic.shape[1],self.orthomosaic.shape[2]))
            self.orthomosaic = vstack((self.orthomosaic, newRows))
            newColumns = zeros((self.orthomosaic.shape[0],int(ceil((self.orthomosaic.shape[1]-self.interval)/(self.interval/2+self.stride)))*int(self.interval/2+self.stride)+self.interval-self.orthomosaic.shape[1],self.orthomosaic.shape[2]))
            self.orthomosaic = hstack((self.orthomosaic, newColumns)).astype('uint8')
            save('Data/'+self.imageryType+'Orthomosaic', self.orthomosaic)
        
        print(self.imageryType + ' orthomosaic shape: ' + str(self.orthomosaic.shape))
        
    def plotImage(self, imageData, title='Figure'):
        figure(title, dpi=150)

        if self.imageryType == 'Thermal':
            imagePlot = imshow(imageData)
            imagePlot.set_cmap('plasma')
            cb = colorbar()
            cb.set_label('Pixel value')

        elif self.imageryType == 'RGB':
            imshow(imageData)

        xlabel('X (pixels)')
        ylabel('Y (pixels)')
        savefig('Images/' + self.imageryType + 'Images/' + title + '.png')
        show()
    
    def getMiddenLocs(self, middenLocationsPath):
        xOrigin, pixelWidth, _, yOrigin, _, pixelHeight = self.dataset.GetGeoTransform()
        self.middenCoords = load(middenLocationsPath).T # in meters
        print(len(self.middenCoords.T))
        self.middenCoords[0] = (self.middenCoords[0]-xOrigin)/pixelWidth - self.startCol # in pixels
        self.middenCoords[1] = (self.middenCoords[1]-yOrigin)/pixelHeight - self.startRow # in pixels
        self.middenCoords = around(self.middenCoords).astype(int)
        self.middenLocsInOrthomosaic = zeros((self.orthomosaic.shape[0],self.orthomosaic.shape[1])).astype(int)

        for index in range(len(self.middenCoords.T)):
            if self.middenCoords.T[index][1] >= self.orthomosaic.shape[0]:
                print(index, self.middenCoords.T[index])
            if self.middenCoords.T[index][0] >= self.orthomosaic.shape[1]:
                print(index, self.middenCoords.T[index])

        for loc in self.middenCoords.T:
            self.middenLocsInOrthomosaic[loc[1],loc[0]] = 1

        if self.imageryType == 'RGB':
            save('Data/'+self.imageryType+'MiddenLocsInOrthomosaic', self.middenLocsInOrthomosaic)

        print(self.imageryType + ': ' + str(xOrigin) + ' ' + str(yOrigin) + ' ' + str(pixelWidth) + ' ' + str(pixelHeight))
        print(self.imageryType + ' num of middens ' + str(sum(self.middenLocsInOrthomosaic)))
    
    # def normalizeBand(self, image):
    #     image -= amin(image) # set minimum to 0

    #     if amax(image) != 0:
    #         image /= amax(image) # set maximum to 1

    #     image = (image-0.5)/0.5 # set pixel values to be between -1 and 1
    #     return image
    
    # def normalizeImage(self, image):
    #     if len(image.shape) == 2:
    #         image = self.normalizeBand(image)

    #     elif len(image.shape) == 3:
    #         normalizedImage = zeros((image.shape))

    #         for band in range(image.shape[0]):
    #             normalizedImage[band,:,:] = self.normalizeBand(image[band,:,:])

    #         image = normalizedImage
    #     return image

    def generateTrainingData(self):
        top = 0
        bottom = self.stride + self.interval/2 + self.stride
        
        while bottom < int(self.orthomosaic.shape[0]):
            left = 0
            right = self.stride + self.interval/2 + self.stride

            while right < int(self.orthomosaic.shape[1]):
                image = self.orthomosaic[int(top):int(bottom),int(left):int(right)].copy()

                if len(image.shape) == 2:
                    image -= amin(image) # set the minimum pixel value to 0
                
                self.trainingImages.append(image)
                self.trainingLabelMatrices.append(self.middenLocsInOrthomosaic[int(top):int(bottom),int(left):int(right)])
                left += self.stride + self.interval/2
                right += self.interval/2 + self.stride

            top += self.stride + self.interval/2
            bottom += self.interval/2 + self.stride

        for i in flip(arange(len(self.trainingImages))):
            if all(self.trainingImages[i]==0):
                del self.trainingImages[i] # remove images whose entries are all 0
                del self.trainingLabelMatrices[i] # remove label matrices whose corresponding images have all zeros
        
        self.trainingLabels = sum(sum(self.trainingLabelMatrices,axis=1),axis=1) # collapses each label matrix to 1 if there's a 1 in the matrix or 0 otherwise
        
        for index in range(len(self.trainingLabels)):
            if self.trainingLabels[index] > 1:
                self.trainingLabels[index] = 1
        
        '''Data Augmentation'''
        # originalLabels = self.trainingLabels.copy()
        # for index in range(len(originalLabels)):
        #     if originalLabels[index] > 0:
        #         self.trainingImages.append(ndimage.rotate(self.trainingImages[index],90))
        #         self.trainingImages.append(ndimage.rotate(self.trainingImages[index],180))
        #         self.trainingImages.append(ndimage.rotate(self.trainingImages[index],270))
        #         self.rotatedImages.append(ndimage.rotate(self.trainingImages[index],180))
        #         self.trainingLabels = append(self.trainingLabels,1)
        #         self.trainingLabels = append(self.trainingLabels,1)
        #         self.trainingLabels = append(self.trainingLabels,1)

        print('Length of training images = ' + str(array(self.trainingImages).shape))
        print('Length of training labels = ' + str(len(self.trainingLabels)))
        print(self.imageryType + ' sum of training labels ' + str(sum(self.trainingLabels)))
        self.trainingData = array(list(zip(self.trainingImages,self.trainingLabels)),dtype=object) # pairs each training image with its label
        print(self.imageryType + ' data shape: ' + str(self.trainingData.shape))
        save('Data/TrainingImages'+self.imageryType, array(self.trainingImages))
        save('Data/TrainingLabels'+self.imageryType, self.trainingLabels)
    
    def showMiddensOnImage(self):
        middenIndices = where(self.trainingLabels == 1)[0]
        noMiddenIndices = where(self.trainingLabels == 0)[0]

        # for img in self.rotatedImages:
        #     image = imshow(img)
            
        #     if self.imageryType == 'Thermal':
        #         image.set_cmap('plasma')
        #         cb = colorbar()
        #         cb.set_label('Pixel value')
        #     xlabel('X (pixels)')
        #     ylabel('Y (pixels)')
        #     show()

        for index in sample(middenIndices.tolist(),10): # look at 10 random images with middens
        #for index in middenIndices:
            print(index)
            figure(self.imageryType + ' Image (Midden) ' + str(index), dpi=150)
            #image = imshow(self.trainingImages[index])
            
            # if self.imageryType == 'Thermal':
            #     image.set_cmap('plasma')
            #     cb = colorbar()
            #     cb.set_label('Pixel value')

            midden = imshow(ma.masked_less(self.trainingLabelMatrices[index],1))
            midden.set_cmap('inferno')
            xlabel('X (pixels)')
            ylabel('Y (pixels)')
            savefig('Images/' + self.imageryType + 'Images/Image' + str(index) + 'Midden.png')
            #show()
        
        for index in sample(noMiddenIndices.tolist(),10): # look at 10 random images with middens
            figure(self.imageryType + ' Image (No Midden) ' + str(index), dpi=150)
            image = imshow(self.trainingImages[index])
            
            if self.imageryType == 'Thermal':
                image.set_cmap('plasma')
                cb = colorbar()
                cb.set_label('Pixel value')
            #midden = imshow(ma.masked_less(self.trainingLabelMatrices[index],1))
            #midden.set_cmap('inferno')
            xlabel('X (pixels)')
            ylabel('Y (pixels)')
            savefig('Images/' + self.imageryType + 'Images/Image' + str(index) + 'Empty.png')
            #show()

if len(argv)==2:
    DataEngineering(orthomosaicPath='Tiffs/Firestorm3' + argv[1] + '.tif', middenLocationsPath='Data/AllMiddenLocations.npy', imageryType=argv[1])

elif len(argv)==3:
    DataEngineering(orthomosaicPath='Tiffs/Firestorm3' + argv[1] + '.tif', middenLocationsPath='Data/AllMiddenLocations.npy', imageryType=argv[1], part=argv[2])