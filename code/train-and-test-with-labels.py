'''run-cnn by Lucia Gordon'''

# imports
import numpy as np
import random
import os
from sys import argv
from cnn import CNN

# global variables
folder = argv[1]
trials = argv[2] # range of integers ("0-29") or comma-separated list in ascending order ("30,33,37")
save = argv[3] # True ("True") or False ("False")
mode = argv[4] # passive ("p") or active ("a")
modality = argv[5] # thermal ("t"), RGB ("r"), or LiDAR ("l")

if mode == 'a':
    query = argv[6] # random ("r"), uncertainty ("u"), certainty ("c"), disagree ("d"), or multimodAL ("m")
else:
    query = '' # the passive mode has no querying

if '-' in trials: # if a range of integers is provided
    start = int(trials.split('-')[0])
    stop = int(trials.split('-')[1])
    trials = np.arange(start, stop)

elif ',' in trials: # if a list of integers is provided
    trials = [int(trials.split(',')[i]) for i in range(len(trials))]

else: # if a single trial number is provided
    trials = [int(trial)]

setting = mode + '-' + modality + '-' + query # overall setting
labels = np.load(folder+'/data/labels.npy') # image labels

# folders
if not os.path.exists(folder+'/results'):
    os.mkdir(folder+'/results')

if not os.path.exists(folder+'/results/figs'):
    os.mkdir(folder+'/results/figs')

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
for trial in trials:
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
        if os.path.exists('results/'+setting): # check if the setting has a file
            results = list(np.load('results/'+setting)) # load the existing file
            
            if trial > len(results): # if the trial number is larger than the length of results
                results += [[] for i in range(trial+1-len(results))] # add empty lists to results
                
        else: # if the setting does not have a file
            results = [[] for i in range(trial+1)] # initialize an empty results array

        results[trial] = model.results

        np.save(folder+'/results2/'+setting, results)
        
        # if mode == 'a':
        #     np.save('results/'+setting+'/accuracy-vs-labels-'+trial, model.accuracies)
        #     np.save('results/'+setting+'/fraction-middens-found-'+trial, model.fraction_middens_found)