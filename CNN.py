''' CNN by Lucia Gordon '''
from numpy import amax, amin, append, array, around, where, take, min, load, delete, mean, std, zeros, save
from random import sample
from torch import device, cuda, nn, sum, optim, min, max, FloatTensor, no_grad, from_numpy, empty, Tensor, cat
from torchvision.models import vgg16
from torch.utils.data import random_split, DataLoader
from torchsummary import summary
from math import ceil
from sys import argv

class CNN:
    def __init__(self, images, labels, batchSize=8, epochs=30, learningRate=0.0001, learningMode='passive', trial='1'):
        self.model = None
        self.images = load(images)
        self.labels = load(labels)
        #goodMiddenIndices = [916, 1482, 1759, 1760, 1859, 1860, 1943, 1944, 2065, 2581, 2582, 3037, 3489, 3681, 3782, 4217, 4318, 4378, 4379, 4805, 5486, 6673, 6674, 6879, 7044, 7209, 7210, 7356, 7456, 7755, 7756, 8273, 8337, 8338, 8501, 8920, 9438]
        #self.data = sample(list(zip(self.images,self.labels)),100)
        #print(len(goodMiddenIndices))
        self.data = list(zip(self.images,self.labels))
        #goodMiddens = take(self.data, goodMiddenIndices, axis=0)
        allMiddens = []
        for point in self.data:
            if point[1] == 1:
                allMiddens.append(point)
        allEmptyImages = []
        for point in self.data:
            if point[1] == 0:
                allEmptyImages.append(point)
        emptyImages = sample(allEmptyImages, len(allMiddens))
        self.data = allMiddens+emptyImages
        print('Number of images = ' + str(len(self.data)))
        self.images, self.labels = zip(*self.data)
        self.dataLength = len(self.data)
        self.batchSize = batchSize
        self.epochs = epochs
        self.learningRate = learningRate
        self.learningMode = learningMode # passive or active
        self.trial = trial
        self.trainingDataLength = 0
        self.testDataLength = 0
        self.trainingData = None
        self.testData = None
        self.trainingLoader = None
        self.testLoader = None
        self.trainingImages = empty((1, self.images[0].shape[0], self.images[0].shape[1]))
        self.trainingLabels = empty((len(self.labels)))
        self.testLabels = []
        self.classWeights = []
        self.accuracy = 0
        self.class0Accuracy = 0
        self.class1Accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0

        self.useVGG16()
        self.preprocessData()
        self.computeClassWeights()

        if self.learningMode == 'passive':
            self.passiveTrain(self.trainingLoader, self.batchSize)
        
        elif self.learningMode == 'active':
            self.activeTrain()
        self.test()

    def useVGG16(self):
        myDevice = device('cuda' if cuda.is_available() else 'cpu')
        self.model = vgg16(weights='VGG16_Weights.DEFAULT').to(myDevice) # imports a pretrained vgg16 CNN
        lastLayerIndex = len(self.model.classifier)-1
        oldFinalLayer = self.model.classifier.__getitem__(lastLayerIndex)
        newFinalLayers = nn.Sequential(nn.Linear(in_features=oldFinalLayer.in_features,out_features=2,bias=True),nn.Softmax(dim=1)) # changes the output of the last hidden layer to be binary and adds softmax
        self.model.classifier.__setitem__(lastLayerIndex,newFinalLayers)

    def preprocessData(self):        
        self.trainingDataLength = around(0.8*self.dataLength).astype(int) # use 80% of the data for training
        self.testDataLength = self.dataLength - self.trainingDataLength # use 20% of the data for testing
        self.trainingData, self.testData = random_split(self.data,[self.trainingDataLength,self.testDataLength]) # splits the data into training and test sets
        
        for _, point in enumerate(self.trainingData):
            self.trainingImages = cat((self.trainingImages, Tensor(array([point[0]]))))
            self.trainingLabels = cat((self.trainingLabels, Tensor(array([point[1]]))))
       
        self.trainingLoader = DataLoader(self.trainingData, batch_size=self.batchSize, shuffle=True, num_workers=1) # batches and shuffles the training data
        self.testLoader = DataLoader(self.testData, batch_size=self.batchSize, shuffle=True, num_workers=1) # batches and shuffles the test data

    def computeClassWeights(self):
        numClass1 = 0

        for _,batch in enumerate(self.trainingLoader):
            numClass1 += batch[1].sum().item() # sum of labels in batch
        
        self.classWeights = [numClass1/self.trainingDataLength, 1 - numClass1/self.trainingDataLength] # the weights are proportional to the number of points in the other class and sum to 1
        print(self.classWeights)
        numClass1Test = 0

        for _,batch in enumerate(self.testLoader):
            numClass1Test += batch[1].sum().item() # sum of labels in batch
        
        print([numClass1Test/self.testDataLength, 1 - numClass1Test/self.testDataLength])

    def normalizeBand(self, image):
        image -= min(image) # set minimum to 0

        if max(image) != 0:
            image /= max(image) # set maximum to 1

        image = (image-0.5)/0.5 # set pixel values to be between -1 and 1
        return image
    
    def normalizeImage(self, image):
        if len(image.shape) == 2:
            image = self.normalizeBand(image)

        elif len(image.shape) == 3:
            normalizedImage = zeros((image.shape[2], image.shape[0], image.shape[1]))

            for band in range(image.shape[0]):
                normalizedImage[band,:,:] = self.normalizeBand(image[:,:,band])

            image = normalizedImage
        return image
    
    def transformImages(self, images):
        transformedImages = empty((images.shape[0],3,images.shape[1],images.shape[2]))
        
        for i in range(len(images)):
            image = self.normalizeImage(images[i])

            if len(image.shape) == 2:
                addedDim = Tensor(array([image.numpy()]))
                tripled = cat((addedDim, addedDim, addedDim))
                transformedImages[i] = tripled
                        
        return transformedImages

    def passiveTrain(self, trainingLoader, batchSize):
        criterion = nn.CrossEntropyLoss(weight=FloatTensor(self.classWeights))
        optimizer = optim.Adam(self.model.parameters(),lr=self.learningRate)
        interval = ceil(len(trainingLoader)/self.batchSize/10)
        patience = 0
        previousLoss = 0

        for epoch in range(self.epochs):  # loop over the dataset multiple times
            runningLoss = 0.0
            totalLoss = 0

            for i, data in enumerate(trainingLoader,0): # data is a list of [images,labels]
                print(data) # FIX THE SHAPE OF DATA
                images, labels = data
                optimizer.zero_grad() # zero the parameter gradients
                outputs = self.model(self.transformImages(images)) # forward pass
                loss = criterion(outputs,labels)
                loss.backward() # backward pass
                optimizer.step() # optimization
                runningLoss += loss.item()
                totalLoss += loss.item()

                if i % interval == interval-1:
                    #print(f'Epoch={epoch+1}, Images {self.batchSize*(i+1-interval)}-{self.batchSize*(i+1)}, Loss={runningLoss/interval:.3f}') # average loss per batch
                    runningLoss = 0.0

            print('Epoch ' + str(epoch+1) + ' loss = ' + str(totalLoss))

            if epoch > 1:
                if totalLoss >= previousLoss:
                    patience += 1
                else:
                    patience = 0
            
            previousLoss = totalLoss
            
            print('Patience = ' + str(patience))
            
            if patience == 5:
                break
        
        print('Finished Training')
    
    def activeTrain(self):
        unlabeledImages = self.trainingImages # all the training images start out unlabeled
        imageLabels = self.trainingLabels
        maxPixelVals = array([max(unlabeledImages[i]) for i in range(len(unlabeledImages))]) # array of maximum pixel value in each image
        labelingBudget = int(around(self.trainingDataLength/10)) # number of images we are willing to provide labels for
        e = 0.01

        def brightestIndices(fraction):
            return sorted(range(len(maxPixelVals)), key=lambda sub:maxPixelVals[sub])[-int(around(fraction*len(maxPixelVals))):]
        
        trainingImageIndices = sample(brightestIndices(e), self.batchSize) # randomly picking a batch among the images with the highest max pixel values
        print('Len training img indices = ',len(trainingImageIndices))
        trainingLoader = [list(take(unlabeledImages, trainingImageIndices, axis=0))]+[list(take(imageLabels, trainingImageIndices, axis=0))] # select the images corresponding to the above indices
        #print(trainingLoader)
        def removeFromUnlabeledImgs():
            for index in sorted(trainingImageIndices, reverse=True):
                delete(unlabeledImages,index) # remove image that will be used for training from the unlabeled set
                delete(imageLabels,index)
                delete(maxPixelVals,index) # remove index that will be used for training from the list of max pixel values

        removeFromUnlabeledImgs()

        while labelingBudget > 0:
            print('Labeling budget = ',labelingBudget)
            print('Training loader len = ', len(trainingLoader))
            #print(trainingLoader[0])
            self.passiveTrain(trainingLoader, 1)
            testImageIndices = brightestIndices(2*e)
            print('Len test img indices = ',len(testImageIndices))
            with no_grad():
                sigmoidOutput = self.model(self.transformImages(take(unlabeledImages,testImageIndices,axis=0))).data.T[1]
            trainingImageIndices = sorted(range(len(sigmoidOutput)), key=lambda sub:sigmoidOutput[sub])[-self.batchSize:]
            nextImages = take(unlabeledImages, trainingImageIndices, axis=0) # do the indices in the above line correspond to these?
            nextLabels = take(imageLabels, trainingImageIndices, axis=0)
            trainingLoader += list(zip(nextImages,nextLabels))
            #trainingLoader = append(trainingLoader,list(zip(nextImages,nextLabels)))
            removeFromUnlabeledImgs()
            labelingBudget -= self.batchSize
            e = amin([2*e,0.5])

    def test(self):
        correct = 0
        class0Correct = 0
        class1Correct = 0
        total = 0
        class0Total = 0
        class1Total = 0
        predictedPositives = 0

        with no_grad(): # since we're not training, we don't need to calculate the gradients for our outputs
            for i, data in enumerate(self.testLoader):
                images, labels = data
                outputs = self.model(self.transformImages(images))
                _, predicted = max(outputs.data, 1) # predicted is a vector with batchSize elements corresponding to the index of the most likely class of each image in the batch
                total += labels.size(0) # number of images in the batch
                correct += (predicted == labels).sum().item() # number of images classified correctly
                class0Indices = where(labels == 0)[0] # indices of no midden images
                class0Total += len(class0Indices) # number of images with no middens
                class0Correct += (take(predicted,class0Indices) == take(labels,class0Indices)).sum().item() # number of true negatives
                class1Indices = where(labels == 1)[0] # indices of midden images
                class1Total += len(class1Indices) # number of images with middens
                class1Correct += (take(predicted,class1Indices) == take(labels,class1Indices)).sum().item() # number of true positives
                predictedPositives += predicted.sum().item() # true positives + false positives

        self.accuracy = round(100*correct/total,3)
        print(f'Accuracy of the neural network on the {self.batchSize*(i+1)} test images = {self.accuracy}%')
        self.class0Accuracy = round(100*class0Correct/class0Total,3)
        print(f'Accuracy on images without middens = {self.class0Accuracy}%')
        self.class1Accuracy = round(100*class1Correct/class1Total,3)
        print(f'Accuracy on images with middens = {self.class1Accuracy}%')

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
        save('Results/Test Results: Trial '+self.trial, array([self.accuracy, self.class0Accuracy, self.class1Accuracy, self.precision, self.recall, self.f1]))

if argv[1] == 'Thermal':
    CNN('Data/TrainingImagesThermal.npy', 'Data/TrainingLabelsThermal.npy', batchSize=8, epochs=int(argv[3]), learningRate=0.0001, learningMode=argv[2])

elif argv[1] == 'RGB':
    CNN('Data/TrainingImagesRGB.npy', 'Data/TrainingLabelsRGB.npy', batchSize=8, epochs=int(argv[3]), learningRate=0.0001, learningMode=argv[2])

#ThermalCNN = CNN('Data/TrainingImagesCIFAR.npy', 'Data/TrainingLabelsCIFAR.npy', batchSize=8, epochs=int(argv[2]), learningRate=0.0001, learningMode=argv[1])

# for i in range(1,6):
#     CNN('Data/TrainingImagesThermal.npy', 'Data/TrainingLabelsThermal.npy', batchSize=8, epochs=int(argv[2]), learningRate=0.0001, learningMode='passive', trial=str(i))