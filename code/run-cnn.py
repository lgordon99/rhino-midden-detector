'''run-cnn by Lucia Gordon'''

# imports
import numpy as np
import random
import os
from sys import argv
from cnn import CNN

# global variables
num_trials = argv[1] # number of trials
save = argv[2] # True ("True") or False ("False")
mode = argv[3] # passive ("p") or active ("a")
modality = argv[4] # thermal ("t"), RGB ("r"), or LiDAR ("l")

if mode == 'a':
    query = argv[5] # random ("r"), uncertainty ("u"), certainty ("c"), disagree ("d"), or multimodAL ("m")
else:
    query = '' # the passive mode has no querying

setting = mode + '-' + modality + '-' + query # overall setting
labels = np.load('data/labels.npy') # image labels

# folders
if not os.path.exists('results'):
    os.mkdir('results')

if not os.path.exists('results/figs'):
    os.mkdir('results/figs')

if not os.path.exists('results/'+setting): # check if the setting has a folder already
    os.mkdir('results/'+setting) # make a folder for the setting

# functions
def train_test_split(middens_empty_ratio_training=None): # select images for training and testing
    num_midden_imgs = np.sum(labels) # total number of images with middens
    test_indices = np.append(random.sample(list(np.where(labels==1)[0]), round(0.2*num_midden_imgs)), random.sample(list(np.where(labels==0)[0]), round(0.2*num_midden_imgs))) # select equal numbers of midden and empty images for testing
    train_indices = list(np.delete(list(range(len(labels))), test_indices)) # all the images not set aside for testing are for training
        
    if middens_empty_ratio_training is not None: # if we want to downsample the empty images in training
        incomplete = True # downsampling process is incomplete

        while incomplete: # while the downsampling process is not complete
            if np.sum(np.take(labels, train_indices)) < middens_empty_ratio_training*len(train_indices)/(middens_empty_ratio_training+1): # if the fraction of middens in the training set is too low
                del train_indices[random.choice(np.where(np.take(labels, train_indices)==0)[0])] # delete an empty image from the training set
            else:
                incomplete = False # downsampling process is complete

    print(len(train_indices))
    print(np.sum(np.take(labels, train_indices)))
    print(np.sum(np.take(labels, test_indices)))
    print(len(test_indices))

    return train_indices, test_indices # the training and test indices can be used to extract the images, labels, and identifiers of the training and test images, respectively

# run model
for trial in range(int(num_trials)):
    trial = str(trial)
    print('Setting:', setting)
    print('Trial ' + trial)

    if mode[0] == 'p': # if we are in the passive mode
        middens_empty_ratio_training = 1 # want equally many empty images as midden images in training
    else: # if we are in the active mode
        middens_empty_ratio_training = None # use the entire training set

    train_indices, test_indices = train_test_split(middens_empty_ratio_training) # randomly select the train and test indices
    model = CNN(mode, modality, query, train_indices, test_indices) # run the model

    if save == 'True': # if we want to save the results
        np.save('results/'+setting+'/results-'+trial, model.results)
        
        if mode == 'a':
            np.save('results/'+setting+'/accuracy-vs-labels-'+trial, model.accuracies)
            np.save('results/'+setting+'/fraction-middens-found'+trial, model.fraction_middens_found)