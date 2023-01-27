''' CNN by Lucia Gordon '''
from numpy import amax, amin, append, array, sum, around, where, take, load, delete, mean, std, save, arange, zeros, argsort, sort, flip
from random import sample, choices
from torch import device, cuda, nn, optim, min, max, FloatTensor, no_grad, from_numpy, empty, Tensor, float64, tensor, ones
from torchvision.models import vgg16
from torchvision.transforms import transforms
from torchsummary import summary
from math import ceil, floor

class CNN:
    def __init__(self, setting, indices):
        # global variables
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
        self.labelingBudget = 300
        self.accuracy = 0
        self.class0Accuracy = 0
        self.class1Accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0
        self.accuracies = []
        self.results = []
        self.trainingMiddenIndices, self.testMiddenIndices, self.trainingEmptyIndices, self.testEmptyIndices = indices
        
        # local variables
        passiveBatchSize = 20
        activeBatchSize = 10
        epochs = 30

        # import images
        thermalTrainingImages, thermalTestImages = self.importImages('Thermal')
        RGBTrainingImages, RGBTestImages = self.importImages('RGB')
        LiDARTrainingImages, LiDARTestImages = self.importImages('LiDAR')
        fusedTrainingImages, fusedTestImages = self.importImages('Fused')

        # import bins
        maxOccupiedBinMiddens = load('Phase3/Data/Thermal/MaxOccupiedBinMiddens.npy')
        maxOccupiedBinEmpty = load('Phase3/Data/Thermal/MaxOccupiedBinEmpty.npy')
        maxTrainingOccupiedBinMiddens = take(maxOccupiedBinMiddens, self.trainingMiddenIndices)
        maxTrainingOccupiedBinEmpty = take(maxOccupiedBinEmpty, self.trainingEmptyIndices)
        maxOccupiedBin = append(maxTrainingOccupiedBinMiddens, maxTrainingOccupiedBinEmpty)

        # initialize models
        thermalModel = self.initializeModel()
        RGBModel = self.initializeModel()
        LiDARModel = self.initializeModel()
        fusedModel = self.initializeModel()

        # generate training loaders
        print('Thermal')
        thermalTrainingLoader = self.makeTrainingLoader(passiveBatchSize, thermalTrainingImages)
        print('RGB')
        RGBTrainingLoader = self.makeTrainingLoader(passiveBatchSize, RGBTrainingImages)
        print('LiDAR')
        LiDARTrainingLoader = self.makeTrainingLoader(passiveBatchSize, LiDARTrainingImages)
        print('Fused')
        fusedTrainingLoader = self.makeTrainingLoader(passiveBatchSize, fusedTrainingImages)

        # generate test loaders
        thermalTestLoader = self.makeTestLoader(passiveBatchSize, thermalTestImages)
        RGBTestLoader = self.makeTestLoader(passiveBatchSize, RGBTestImages)
        LiDARTestLoader = self.makeTestLoader(passiveBatchSize, LiDARTestImages)
        fusedTestLoader = self.makeTestLoader(passiveBatchSize, fusedTestImages)

        # train model
        if setting == 'PassiveThermal':
            self.passiveTrain(thermalModel, thermalTrainingLoader, epochs, setting, thermalTestLoader)

        elif setting == 'PassiveRGB':
            self.passiveTrain(RGBModel, RGBTrainingLoader, epochs, setting, RGBTestLoader)
        
        elif setting == 'PassiveLiDAR':
            self.passiveTrain(LiDARModel, LiDARTrainingLoader, epochs, setting, LiDARTestLoader)

        elif setting == 'PassiveFused':
            self.passiveTrain(fusedModel, fusedTrainingLoader, epochs, setting, fusedTestLoader)
        
        elif setting == 'ActiveUncertainty':
            self.uncertaintyActiveTrain(activeBatchSize, thermalModel, thermalTrainingImages, thermalTestLoader, 'True')

        elif setting == 'ActiveCertainty':
            self.uncertaintyActiveTrain(activeBatchSize, thermalModel, thermalTrainingImages, thermalTestLoader, 'False')

        elif setting == 'ActiveBinningThermal':
            self.binningActiveTrain(activeBatchSize, maxOccupiedBin, thermalModel, thermalTrainingImages, thermalTestLoader)
        
        elif setting == 'ActiveBinningFused':
            self.binningActiveTrain(activeBatchSize, maxOccupiedBin, fusedModel, fusedTrainingImages, fusedTestLoader)

        elif setting == 'ActiveBinningThermalRGB':
            self.binningActiveTrain(activeBatchSize, maxOccupiedBin, thermalModel, thermalTrainingImages, thermalTestLoader, RGBModel, RGBTrainingImages, RGBTestLoader)

        elif setting == 'ActiveBinningThermalRGBLiDAR':
            self.binningActiveTrain(activeBatchSize, maxOccupiedBin, thermalModel, thermalTrainingImages, thermalTestLoader, RGBModel, RGBTrainingImages, RGBTestLoader, LiDARModel, LiDARTrainingImages, LiDARTestLoader)

        elif setting == 'ActiveDisagree':
            self.disagreeActiveTrain(activeBatchSize, thermalModel, thermalTrainingImages, thermalTestLoader, RGBModel, RGBTrainingImages, RGBTestLoader)

    def importImages(self, modality):
        middenImages = load('Phase3/Data/'+modality+'/MiddenImages.npy')
        rotated90MiddenImages = load('Phase3/Data/'+modality+'/Rotated90MiddenImages.npy')
        rotated180MiddenImages = load('Phase3/Data/'+modality+'/Rotated180MiddenImages.npy')
        rotated270MiddenImages = load('Phase3/Data/'+modality+'/Rotated270MiddenImages.npy')
        emptyImages = load('Phase3/Data/'+modality+'/EmptyImages.npy')

        trainingMiddenImages = take(middenImages, self.trainingMiddenIndices, axis=0)
        trainingRotated90MiddenImages = take(rotated90MiddenImages, self.trainingMiddenIndices, axis=0)
        trainingRotated180MiddenImages = take(rotated180MiddenImages, self.trainingMiddenIndices, axis=0)
        trainingRotated270MiddenImages = take(rotated270MiddenImages, self.trainingMiddenIndices, axis=0)
        trainingEmptyImages = take(emptyImages, self.trainingEmptyIndices, axis=0)
        trainingImages = [trainingMiddenImages, trainingRotated90MiddenImages, trainingRotated180MiddenImages, trainingRotated270MiddenImages, trainingEmptyImages]

        testMiddenImages = take(middenImages, self.testMiddenIndices, axis=0)
        testEmptyImages = take(emptyImages, self.testEmptyIndices, axis=0)
        testImages = [testMiddenImages, testEmptyImages]

        return trainingImages, testImages

    def initializeModel(self):
        myDevice = device('cuda' if cuda.is_available() else 'cpu')
        model = vgg16(weights='VGG16_Weights.DEFAULT').to(myDevice) # imports a pretrained vgg16 CNN

        for parameter in model.parameters(): # freeze all parameters
            parameter.requires_grad = False
        
        model.classifier = nn.Sequential(nn.Linear(in_features=25088, out_features=512),
                                            nn.ReLU(),
                                            nn.Dropout(),
                                            nn.Linear(in_features=512, out_features=256),
                                            nn.ReLU(),
                                            nn.Dropout(),
                                            nn.Linear(in_features=256, out_features=128),
                                            nn.ReLU(),
                                            nn.Dropout(),
                                            nn.Linear(in_features=128, out_features=1),
                                            nn.Sigmoid()) # unfreeze classifier parameters
        return model

    def makeTrainingLoader(self, batchSize, trainingImages):
        allTrainingMiddens = append(trainingImages[0], append(trainingImages[1], append(trainingImages[2], trainingImages[3], axis=0), axis=0), axis=0)        
        print(len(allTrainingMiddens), ' training midden images')
        trainingEmpty = trainingImages[4]
        print(len(trainingEmpty), ' training empty images')
        allTrainingImages = append(allTrainingMiddens, trainingEmpty, axis=0) # combine midden and empty images for training
        trainingData = list(zip(allTrainingImages,[1]*len(allTrainingMiddens)+[0]*len(trainingEmpty))) # add labels to the training images
        trainingData = sample(trainingData, len(trainingData)) # shuffle the training images
        trainingLoader = []
        images = []
        labels = []

        # batch the training data
        for i in range(len(trainingData)+1): 
            if (i % batchSize == 0 and i != 0) or (i == len(trainingData)):
                trainingLoader.append([self.transformImages(array(images)), tensor(labels).to(float64)])
                images = []
                labels = []
            
            if i != len(trainingData):
                images.append(trainingData[i][0])
                labels.append(trainingData[i][1])
        
        return trainingLoader

    def makeTestLoader(self, batchSize, testImages):
        testMiddens = testImages[0]
        testEmpty = testImages[1]
        allTestImages = append(testMiddens, testEmpty, axis=0) # combine midden and empty images for testing
        testData = list(zip(allTestImages,[1]*len(testMiddens)+[0]*len(testEmpty))) # add labels to the test images
        testData = sample(testData, len(testData)) # shuffle the test images
        testLoader = []
        images = []
        labels = []

        # batch the test data
        for i in range(len(testData)+1): 
            if (i % batchSize == 0 and i != 0) or (i == len(testData)):
                testLoader.append([self.transformImages(array(images)), tensor(labels).to(float64)])
                images = []
                labels = []
            
            if i != len(testData):
                images.append(testData[i][0])
                labels.append(testData[i][1])
        
        return testLoader
    
    def transformImages(self, images):
        transformedImages = empty((images.shape[0],images.shape[3],images.shape[1],images.shape[2])) # (batchSize, 3, 224, 224)
        
        for i in range(len(images)):
            transformedImages[i] = self.transform(images[i])
                        
        return transformedImages

    def passiveTrain(self, model, trainingLoader, epochs, setting, testLoader=None):
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        patience = 0
        previousLoss = 0

        for epoch in range(epochs):  # loop over the dataset multiple times
            totalLoss = 0

            for _, data in enumerate(trainingLoader): # data is a list of [images, labels]
                images, labels = data
                optimizer.zero_grad() # zero the parameter gradients
                outputs = model(images).flatten().to(float64) # forward pass
                loss = criterion(outputs, labels)
                loss.backward() # backward pass
                optimizer.step() # optimization
                totalLoss += loss.item()

            print('Epoch ' + str(epoch+1) + ' loss = ' + str(round(totalLoss,3)))

            if epoch > 1:
                if totalLoss >= previousLoss:
                    patience += 1
                else:
                    patience = 0
            
            previousLoss = totalLoss
            print('Patience = ' + str(patience))
            
            # if patience == 5:
            #     break
            
            if setting[0] == 'P':
                self.test(model, testLoader)
                self.accuracies.append(self.accuracy)

    def uncertaintyActiveTrain(self, batchSize, model, trainingImages, testLoader, uncertainty):
        trainingMiddens, trainingRotated90Middens, trainingRotated180Middens, trainingRotated270Middens, trainingEmpty = trainingImages
        originalTrainingImages = append(trainingMiddens, trainingEmpty, axis=0)        
        imageLabels = [1]*len(trainingMiddens)+[0]*len(trainingEmpty)
        trainingLoader = []
        usedIndices = []
        unusedIndices = arange(len(originalTrainingImages))
        imagesLabeled = 0
        print('Uncertainty =', uncertainty)

        while imagesLabeled < self.labelingBudget:
            print('Number of used indices =', len(usedIndices))
            print('Number of unused indices =', len(unusedIndices))
            print('Number of indices =', len(usedIndices)+len(unusedIndices))

            imagesLeft = self.labelingBudget - imagesLabeled
            indicesToPredictOn = sample(list(unusedIndices), self.labelingBudget)

            with no_grad():
                outputs = model(self.transformImages(take(originalTrainingImages, indicesToPredictOn, axis=0))).flatten().cpu().detach().numpy()
                print('Number of images predicted on =', len(outputs))
            
            if uncertainty:
                selectedIndices = take(indicesToPredictOn, argsort(abs(outputs-0.5))[:amin([batchSize, imagesLeft])])
            else:
                selectedIndices = take(indicesToPredictOn, flip(argsort(outputs))[:amin([batchSize, imagesLeft])])
            
            print('Number of selected indices =', len(selectedIndices))
            usedIndices += list(selectedIndices)

            for i in reversed(range(len(unusedIndices))):
                if unusedIndices[i] in usedIndices:
                    unusedIndices = delete(unusedIndices, i)

            trainingIndices = [] # the indices for the images that will be used in the next round of training
            additionalMiddenIndices = []

            for index in selectedIndices:
                if imageLabels[index] == 1: # if a midden has been selected
                    trainingIndices.append(index) # include it in the images for training
            
            numNewMiddens = len(trainingIndices) # number of middens found in this search
            print('Number of new middens =', numNewMiddens)
            totalMiddensFound = sum(take(imageLabels, usedIndices)) # total number of middens found so far
            print('Total number of middens found =', totalMiddensFound)

            if totalMiddensFound > 0:            
                if numNewMiddens < batchSize/2: # for batchSize=10, this checks if at least 5 middens have been found
                    additionalMiddenIndices = choices(list(take(usedIndices, where(take(imageLabels, usedIndices) == 1)[0])), k=int(batchSize/2-numNewMiddens)) # reuses however many middens are needed to balance the batch

            trainingIndices += additionalMiddenIndices # at this point trainingIndices should be at least half middens

            for index in selectedIndices:
                if (imageLabels[index] == 0) and (len(trainingIndices) < batchSize): # if the index has label 0 and the batch is not full
                    trainingIndices.append(index) # add the index corresponding to an empty image to the batch

            nextTrainingThermalImages = self.transformImages(take(originalTrainingImages, trainingIndices, axis=0))
            trainingLabels = tensor(take(imageLabels, trainingIndices)).to(float64)
            trainingLoader += [[nextTrainingThermalImages]+[trainingLabels]] # select the images corresponding to the above indices

            for batch in trainingLoader:
                print('Number of middens in batch =', int(batch[1].sum().item()))
            
            imagesLabeled += len(selectedIndices)
            print('Images labeled =', imagesLabeled)
            fractionMiddens = totalMiddensFound/imagesLabeled
            print('Fraction of training images with middens =', round(fractionMiddens,3))
            self.passiveTrain(model, trainingLoader, 2, 'active')
            
            if (imagesLabeled < self.labelingBudget) or (totalMiddensFound == 0):
                self.test(model, testLoader)
                self.accuracies.append(self.accuracy)

        if totalMiddensFound > 0:
            middenIndices = []
            emptyIndices = []

            for index in usedIndices:
                if imageLabels[index] == 1:
                    middenIndices.append(index)

            for index in usedIndices:
                if imageLabels[index] == 0:
                    emptyIndices.append(index)

            foundMiddens = take(originalTrainingImages, middenIndices, axis=0)
            foundRotated90Middens = take(trainingRotated90Middens, middenIndices, axis=0)
            foundRotated180Middens = take(trainingRotated180Middens, middenIndices, axis=0)
            foundRotated270Middens = take(trainingRotated270Middens, middenIndices, axis=0)
            allFoundMiddens = append(foundMiddens, append(foundRotated90Middens, append(foundRotated180Middens, foundRotated270Middens, axis=0), axis=0), axis=0)
            print(len(allFoundMiddens))
            subsetEmptyIndices = sample(emptyIndices, 4*len(middenIndices))
            subsetFoundEmpty = take(originalTrainingImages, subsetEmptyIndices, axis=0)
            print(len(subsetFoundEmpty))
            subsetFoundImages = append(allFoundMiddens, subsetFoundEmpty, axis=0) # combine midden and empty images for training
            trainingData = list(zip(subsetFoundImages,[1]*len(allFoundMiddens)+[0]*len(subsetFoundEmpty))) # add labels to the training images
            trainingData = sample(trainingData, len(trainingData)) # shuffle the training images
            trainingLoader = []
            images = []
            labels = []

            for i in range(len(trainingData)+1): 
                if (i % batchSize == 0 and i != 0) or (i == len(trainingData)):
                    trainingLoader.append([self.transformImages(array(images)), tensor(labels).to(float64)])
                    images = []
                    labels = []
                
                if i != len(trainingData):
                    images.append(trainingData[i][0])
                    labels.append(trainingData[i][1])
            
            self.passiveTrain(model, trainingLoader, 20, 'Passive', testLoader)
                    
    def binningActiveTrain(self, batchSize, maxOccupiedBin, model1, trainingImages1, testLoader1, model2=None, trainingImages2=None, testLoader2=None, model3=None, trainingImages3=None, testLoader3=None):
        trainingMiddens1, trainingRotated90Middens1, trainingRotated180Middens1, trainingRotated270Middens1, trainingEmpty1 = trainingImages1
        originalTrainingImages1 = append(trainingMiddens1, trainingEmpty1, axis=0)        
        
        if model2 is not None:
            trainingMiddens2, _, _, _, trainingEmpty2 = trainingImages2
            originalTrainingImages2 = append(trainingMiddens2, trainingEmpty2, axis=0)        
        
        if model3 is not None:
            trainingMiddens3, _, _, _, trainingEmpty3 = trainingImages3
            originalTrainingImages3 = append(trainingMiddens3, trainingEmpty3, axis=0)                
        
        imageLabels = [1]*len(trainingMiddens1)+[0]*len(trainingEmpty1)
        trainingLoader1 = []
        trainingLoader2 = []
        trainingLoader3 = []
        usedIndices = []
        unusedIndices = arange(len(originalTrainingImages1))
        imagesLabeled = 0
        binThreshold = sort(maxOccupiedBin)[-batchSize:][0] # exactly batchSize number of images are above the initial bin threshold
        binInterval = floor(binThreshold/(self.labelingBudget/batchSize)) # how much should the bin threshold change
        
        while imagesLabeled < self.labelingBudget:
            while len(where(take(maxOccupiedBin, unusedIndices) >= binThreshold)[0]) < batchSize: # if less than batchSize number of images are above the bin threshold
                binThreshold -= binInterval

            print('Bin threshold =', binThreshold)

            numAboveThreshold = len(where(take(maxOccupiedBin, unusedIndices) >= binThreshold)[0])

            print('Number of images above threshold =', numAboveThreshold)
            print('Number of used indices =', len(usedIndices))
            print('Number of unused indices =', len(unusedIndices))
            print('Number of indices =', len(usedIndices)+len(unusedIndices))

            imagesLeft = self.labelingBudget - imagesLabeled
            indicesToPredictOn = take(unusedIndices, sample(list(where(take(maxOccupiedBin, unusedIndices) >= binThreshold)[0]), amin([numAboveThreshold, self.labelingBudget])))

            with no_grad():
                outputs1 = model1(self.transformImages(take(originalTrainingImages1, indicesToPredictOn, axis=0))).flatten().cpu().detach().numpy()
                scores = outputs1

                if model2 is not None:
                    outputs2 = model2(self.transformImages(take(originalTrainingImages2, indicesToPredictOn, axis=0))).flatten().cpu().detach().numpy()
                    scores += outputs2

                if model3 is not None:
                    outputs3 = model3(self.transformImages(take(originalTrainingImages3, indicesToPredictOn, axis=0))).flatten().cpu().detach().numpy()
                    scores += outputs3
                
                print('Number of images predicted on =', len(scores))

            selectedIndices = take(indicesToPredictOn, flip(argsort(scores))[:amin([batchSize, imagesLeft])])
            print('Number of selected indices =', len(selectedIndices))
            usedIndices += list(selectedIndices)

            for i in reversed(range(len(unusedIndices))): # for each index in the unused list
                if unusedIndices[i] in usedIndices: # if the index appears in the used list
                    unusedIndices = delete(unusedIndices, i) # remove the index from the unused list

            trainingIndices = [] # the indices for the images that will be used in the next round of training
            additionalMiddenIndices = []

            for index in selectedIndices:
                if imageLabels[index] == 1: # if a midden has been selected
                    trainingIndices.append(index) # include it in the images for training
            
            numNewMiddens = len(trainingIndices) # number of middens found in this search
            print('Number of new middens =', numNewMiddens)
            totalMiddensFound = sum(take(imageLabels, usedIndices)) # total number of middens found so far
            print('Total number of middens found =', totalMiddensFound)

            if numNewMiddens >= batchSize/4: # for batchSize=10, this checks if at least 3 middens have been found in this search
                binThreshold -= binInterval # decrease the bin threshold

            elif numNewMiddens == 0: # if no new middens have been found in this search
                binThreshold += binInterval # increase the bin threshold
            
            if numNewMiddens < batchSize/2: # for batchSize=10, this checks if at least 5 middens have been found
                additionalMiddenIndices = sample(list(take(usedIndices, where(take(imageLabels, usedIndices) == 1)[0])), int(batchSize/2-numNewMiddens)) # reuses however many middens are needed to balance the batch

            trainingIndices += additionalMiddenIndices # at this point trainingIndices should be at least half middens

            for index in selectedIndices:
                if (imageLabels[index] == 0) and (len(trainingIndices) < batchSize): # if the index has label 0 and the batch is not full
                    trainingIndices.append(index) # add the index corresponding to an empty image to the batch

            batchImages1 = self.transformImages(take(originalTrainingImages1, trainingIndices, axis=0))
            if model2 is not None: batchImages2 = self.transformImages(take(originalTrainingImages2, trainingIndices, axis=0))
            if model3 is not None: batchImages3 = self.transformImages(take(originalTrainingImages3, trainingIndices, axis=0))
            batchLabels = tensor(take(imageLabels, trainingIndices)).to(float64)
            trainingLoader1 += [[batchImages1]+[batchLabels]] # select the images corresponding to the above indices
            if model2 is not None: trainingLoader2 += [[batchImages2]+[batchLabels]] # select the images corresponding to the above indices
            if model3 is not None: trainingLoader3 += [[batchImages3]+[batchLabels]] # select the images corresponding to the above indices

            for batch in trainingLoader1:
                print('Number of middens in batch =', int(batch[1].sum().item()))
            
            imagesLabeled += len(trainingIndices)
            print('Images labeled =', imagesLabeled)
            fractionMiddens = totalMiddensFound/imagesLabeled
            print('Fraction of training images with middens =', round(fractionMiddens,3))
            self.passiveTrain(model1, trainingLoader1, 2, 'active')
            if model2 is not None: self.passiveTrain(model2, trainingLoader2, 2, 'active')
            if model3 is not None: self.passiveTrain(model3, trainingLoader3, 2, 'active')

            if imagesLabeled < self.labelingBudget:
                self.test(model1, testLoader1)
                self.accuracies.append(self.accuracy)
        
        middenIndices = [] # indices of found midden images
        emptyIndices = [] # indices of found empty images

        for index in usedIndices:
            if imageLabels[index] == 1:
                middenIndices.append(index)

        for index in usedIndices:
            if imageLabels[index] == 0:
                emptyIndices.append(index)

        foundMiddens = take(originalTrainingImages1, middenIndices, axis=0)
        foundRotated90Middens = take(trainingRotated90Middens1, middenIndices, axis=0)
        foundRotated180Middens = take(trainingRotated180Middens1, middenIndices, axis=0)
        foundRotated270Middens = take(trainingRotated270Middens1, middenIndices, axis=0)
        allFoundMiddens = append(foundMiddens, append(foundRotated90Middens, append(foundRotated180Middens, foundRotated270Middens, axis=0), axis=0), axis=0)
        print(len(allFoundMiddens))
        subsetEmptyIndices = sample(emptyIndices, 4*len(middenIndices))
        subsetFoundEmpty = take(originalTrainingImages1, subsetEmptyIndices, axis=0)
        print(len(subsetFoundEmpty))
        subsetFoundImages = append(allFoundMiddens, subsetFoundEmpty, axis=0) # combine midden and empty images for training
        trainingData = list(zip(subsetFoundImages,[1]*len(allFoundMiddens)+[0]*len(subsetFoundEmpty))) # add labels to the training images
        trainingData = sample(trainingData, len(trainingData)) # shuffle the training images
        trainingLoader = []
        images = []
        labels = []

        for i in range(len(trainingData)+1): 
            if (i % batchSize == 0 and i != 0) or (i == len(trainingData)):
                trainingLoader.append([self.transformImages(array(images)), tensor(labels).to(float64)])
                images = []
                labels = []
            
            if i != len(trainingData):
                images.append(trainingData[i][0])
                labels.append(trainingData[i][1])
        
        self.passiveTrain(model1, trainingLoader, 20, 'passive', testLoader1)
            
    def disagreeActiveTrain(self, batchSize, model1, trainingImages1, testLoader1, model2, trainingImages2, testLoader2):
        trainingMiddens1, trainingRotated90Middens1, trainingRotated180Middens1, trainingRotated270Middens1, trainingEmpty1 = trainingImages1
        originalTrainingImages1 = append(trainingMiddens1, trainingEmpty1, axis=0)        
        trainingMiddens2, _, _, _, trainingEmpty2 = trainingImages2
        originalTrainingImages2 = append(trainingMiddens2, trainingEmpty2, axis=0)        
        imageLabels = [1]*len(trainingMiddens1)+[0]*len(trainingEmpty1)
        trainingLoader1 = []
        trainingLoader2 = []
        usedIndices = []
        unusedIndices = arange(len(originalTrainingImages1))
        imagesLabeled = 0

        while imagesLabeled < self.labelingBudget:
            print('Number of used indices =', len(usedIndices))
            print('Number of unused indices =', len(unusedIndices))
            print('Number of indices =', len(usedIndices)+len(unusedIndices))

            imagesLeft = self.labelingBudget - imagesLabeled
            indicesToPredictOn = sample(list(unusedIndices), self.labelingBudget)

            with no_grad():
                outputs1 = model1(self.transformImages(take(originalTrainingImages1, indicesToPredictOn, axis=0))).flatten().cpu().detach().numpy()
                outputs2 = model2(self.transformImages(take(originalTrainingImages2, indicesToPredictOn, axis=0))).flatten().cpu().detach().numpy()
                scores = abs(outputs1-outputs2)
                print('Number of images predicted on =', len(scores))

            selectedIndices = take(indicesToPredictOn, flip(argsort(scores))[:amin([batchSize, imagesLeft])])
            print('Number of selected indices =', len(selectedIndices))
            usedIndices += list(selectedIndices)

            for i in reversed(range(len(unusedIndices))):
                if unusedIndices[i] in usedIndices:
                    unusedIndices = delete(unusedIndices, i)
            
            trainingIndices = [] # the indices for the images that will be used in the next round of training
            additionalMiddenIndices = []

            for index in selectedIndices:
                if imageLabels[index] == 1: # if a midden has been selected
                    trainingIndices.append(index) # include it in the images for training
            
            numNewMiddens = len(trainingIndices) # number of middens found in this search
            print('Number of new middens =', numNewMiddens)
            totalMiddensFound = sum(take(imageLabels, usedIndices)) # total number of middens found so far
            print('Total number of middens found =', totalMiddensFound)

            if totalMiddensFound > 0:            
                if numNewMiddens < batchSize/2: # for batchSize=10, this checks if at least 5 middens have been found
                    additionalMiddenIndices = choices(list(take(usedIndices, where(take(imageLabels, usedIndices) == 1)[0])), k=int(batchSize/2-numNewMiddens)) # reuses however many middens are needed to balance the batch

            trainingIndices += additionalMiddenIndices # at this point trainingIndices should be at least half middens

            for index in selectedIndices:
                if (imageLabels[index] == 0) and (len(trainingIndices) < batchSize): # if the index has label 0 and the batch is not full
                    trainingIndices.append(index) # add the index corresponding to an empty image to the batch

            batchImages1 = self.transformImages(take(originalTrainingImages1, trainingIndices, axis=0))
            batchImages2 = self.transformImages(take(originalTrainingImages2, trainingIndices, axis=0))
            batchLabels = tensor(take(imageLabels, trainingIndices)).to(float64)
            trainingLoader1 += [[batchImages1]+[batchLabels]] # select the images corresponding to the above indices
            trainingLoader2 += [[batchImages2]+[batchLabels]] # select the images corresponding to the above indices

            for batch in trainingLoader1:
                print('Number of middens in batch =', int(batch[1].sum().item()))
            
            imagesLabeled += len(selectedIndices)
            print('Images labeled =', imagesLabeled)
            fractionMiddens = totalMiddensFound/imagesLabeled
            print('Fraction of training images with middens =', round(fractionMiddens,3))
            self.passiveTrain(model1, trainingLoader1, 2, 'active')
            self.passiveTrain(model2, trainingLoader2, 2, 'active')
            
            if (imagesLabeled < self.labelingBudget) or (totalMiddensFound == 0):
                self.test(model1, testLoader1)
                self.accuracies.append(self.accuracy)

        if totalMiddensFound > 0:
            middenIndices = []
            emptyIndices = []

            for index in usedIndices:
                if imageLabels[index] == 1:
                    middenIndices.append(index)

            for index in usedIndices:
                if imageLabels[index] == 0:
                    emptyIndices.append(index)

            foundMiddens = take(originalTrainingImages1, middenIndices, axis=0)
            foundRotated90Middens = take(trainingRotated90Middens1, middenIndices, axis=0)
            foundRotated180Middens = take(trainingRotated180Middens1, middenIndices, axis=0)
            foundRotated270Middens = take(trainingRotated270Middens1, middenIndices, axis=0)
            allFoundMiddens = append(foundMiddens, append(foundRotated90Middens, append(foundRotated180Middens, foundRotated270Middens, axis=0), axis=0), axis=0)
            print(len(allFoundMiddens))
            subsetEmptyIndices = sample(emptyIndices, 4*len(middenIndices))
            subsetFoundEmpty = take(originalTrainingImages1, subsetEmptyIndices, axis=0)
            print(len(subsetFoundEmpty))
            subsetFoundImages = append(allFoundMiddens, subsetFoundEmpty, axis=0) # combine midden and empty images for training
            trainingData = list(zip(subsetFoundImages,[1]*len(allFoundMiddens)+[0]*len(subsetFoundEmpty))) # add labels to the training images
            trainingData = sample(trainingData, len(trainingData)) # shuffle the training images
            trainingLoader = []
            images = []
            labels = []

            for i in range(len(trainingData)+1): 
                if (i % batchSize == 0 and i != 0) or (i == len(trainingData)):
                    trainingLoader.append([self.transformImages(array(images)), tensor(labels).to(float64)])
                    images = []
                    labels = []
                
                if i != len(trainingData):
                    images.append(trainingData[i][0])
                    labels.append(trainingData[i][1])
            
            self.passiveTrain(model1, trainingLoader, 20, 'Passive', testLoader1)

    def test(self, model, testLoader):
        correct = 0
        # correct2 = 0
        # correct3 = 0
        # correct4 = 0
        # correct5 = 0
        class0Correct = 0
        class1Correct = 0
        total = 0
        class0Total = 0
        class1Total = 0
        predictedPositives = 0

        with no_grad(): # since we're not training, we don't need to calculate the gradients for our outputs
            for i, data in enumerate(testLoader):
                images, labels = data
                outputs = model(images).flatten().to(float64)
                predicted = around(outputs) # > 0.5 --> midden, <= 0.5 --> empty
                # predicted2 = empty(len(predicted))
                # predicted3 = empty(len(predicted))
                # predicted4 = empty(len(predicted))
                # predicted5 = empty(len(predicted))

                # for index in range(len(outputs)):
                #     if outputs[index] > 0.3:
                #         predicted2[index] = 1
                #     else:
                #         predicted2[index] = 0
                    
                #     if outputs[index] > 0.4:
                #         predicted3[index] = 1
                #     else:
                #         predicted3[index] = 0

                #     if outputs[index] > 0.6:
                #         predicted4[index] = 1
                #     else:
                #         predicted4[index] = 0

                #     if outputs[index] > 0.7:
                #         predicted5[index] = 1
                #     else:
                #         predicted5[index] = 0

                total += len(predicted) # number of images in the batch
                correct += (predicted == labels).sum().item() # number of images classified correctly
                # correct2 += (predicted2 == labels).sum().item()
                # correct3 += (predicted3 == labels).sum().item()
                # correct4 += (predicted4 == labels).sum().item()
                # correct5 += (predicted5 == labels).sum().item()
                class0Indices = where(labels == 0)[0] # indices of no midden images
                class0Total += len(class0Indices) # number of images with no middens
                class0Correct += (take(predicted,class0Indices) == take(labels,class0Indices)).sum().item() # number of true negatives
                class1Indices = where(labels == 1)[0] # indices of midden images
                class1Total += len(class1Indices) # number of images with middens
                class1Correct += (take(predicted,class1Indices) == take(labels,class1Indices)).sum().item() # number of true positives
                predictedPositives += predicted.sum().item() # true positives + false positives

        self.accuracy = round(correct/total,3)
        print(f'Accuracy of the neural network on the {total} test images = {self.accuracy}')
        # print(f'Accuracy of the neural network on the {total} test images with 0.3 threshold = {round(correct2/total,3)}')
        # print(f'Accuracy of the neural network on the {total} test images with 0.4 threshold = {round(correct3/total,3)}')
        # print(f'Accuracy of the neural network on the {total} test images with 0.6 threshold = {round(correct4/total,3)}')
        # print(f'Accuracy of the neural network on the {total} test images with 0.7 threshold = {round(correct5/total,3)}')
        self.class0Accuracy = round(class0Correct/class0Total,3)
        print(f'Accuracy on images without middens = {self.class0Accuracy}')
        self.class1Accuracy = round(class1Correct/class1Total,3)
        print(f'Accuracy on images with middens = {self.class1Accuracy}')

        if predictedPositives > 0:
            self.precision = round(class1Correct/predictedPositives,3)
        else:
            self.precision = 0

        print(f'Precision = {self.precision}') # fraction of images classified as having middens that actually have middens

        self.recall = round(class1Correct/class1Total,3)
        print(f'Recall = {self.recall}') # fraction of images with middens classified as having middens
        
        if self.precision == 0 and self.recall == 0:
            self.f1 = 0
        else:
            self.f1 = round(2*self.precision*self.recall/(self.precision+self.recall),3)
        
        print(f'F1 score = {self.f1}') # harmonic mean of precision and recall
        self.results = [self.accuracy, self.class0Accuracy, self.class1Accuracy, self.precision, self.recall, self.f1]