from numpy import amax, amin, append, array, around, where, take, load, delete, mean, std, save, arange
from random import sample
from os import mkdir, path, remove, listdir
from shutil import rmtree

# Set up train-test split: 80% of middens for training, 20% of middens for testing, test on equal number of empty images, train on the rest of the empty images
class TrainTestSplit:
    def __init__(self, setting, trial, balance):
        numMiddenImages = len(load('Phase3/Data/Thermal/MiddenImages.npy'))
        middenIndices = arange(numMiddenImages)
        numTrainingMiddens = around(0.8*numMiddenImages).astype(int)
        trainingMiddenIndices = array(sample(list(middenIndices), numTrainingMiddens))
        testMiddenIndices = middenIndices.copy()

        for i in reversed(range(len(testMiddenIndices))):
            if testMiddenIndices[i] in trainingMiddenIndices:
                testMiddenIndices = delete(testMiddenIndices, i)

        numEmptyImages = len(load('Phase3/Data/Thermal/EmptyImages.npy'))
        emptyIndices = arange(numEmptyImages)
        numTestEmpty = len(testMiddenIndices)
        testEmptyIndices = array(sample(list(emptyIndices), numTestEmpty))
        trainingEmptyIndices = emptyIndices.copy()

        for i in reversed(range(len(trainingEmptyIndices))):
            if trainingEmptyIndices[i] in testEmptyIndices:
                trainingEmptyIndices = delete(trainingEmptyIndices, i)

        if balance:
            trainingEmptyIndices = array(sample(list(trainingEmptyIndices), 4*numTrainingMiddens))
        
        self.indices = [trainingMiddenIndices, testMiddenIndices, trainingEmptyIndices, testEmptyIndices]

        # save('Phase3/Indices/'+setting+'/TrainingMiddenIndices'+trial, trainingMiddenIndices)
        # save('Phase3/Indices/'+setting+'/TestMiddenIndices'+trial, testMiddenIndices)
        # save('Phase3/Indices/'+setting+'/TrainingEmptyIndices'+trial, trainingEmptyIndices)
        # save('Phase3/Indices/'+setting+'/TestEmptyIndices'+trial, testEmptyIndices)