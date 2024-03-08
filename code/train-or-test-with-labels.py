'''train-passive.py by Lucia Gordon'''

# imports
import os
import random
import numpy as np
from sys import argv
from cnn import CNN

# global variables
folder = argv[1] # site name
modality = argv[2] # thermal ("t"), RGB ("r"), or LiDAR ("l")
train = True if argv[3] == 'train' else False # train ("train") or no train ("no_train")
test = True if argv[4] == 'test' else False # test ("test") or no test ("no_test")
labels = np.load(folder+'/data/labels.npy') if train or test else None # image labels

# folders
if not os.path.exists(folder+'/models'):
    os.mkdir(folder+'/models')

# functions
def downsample_empty_images(middens_empty_ratio_training):
    train_indices = list(range(len(labels))) # initially all the images are included for training
    incomplete = True # downsampling process is incomplete

    while incomplete: # while the downsampling process is not complete
        if np.sum(np.take(labels, train_indices)) < middens_empty_ratio_training*len(train_indices)/(middens_empty_ratio_training+1): # if the fraction of middens in the training set is too low
            del train_indices[random.choice(np.where(np.take(labels, train_indices)==0)[0])] # delete an empty image from the training set
        else:
            incomplete = False # downsampling process is complete

    print(len(train_indices), 'training images')
    print(np.sum(np.take(labels, train_indices)), 'training middens')

    return train_indices

train_indices = downsample_empty_images(1) if train else None
test_indices = list(range(len(labels))) if test else None
starting_models = 'vgg16' if train else ['firestorm-3/models/model-'+i for i in modality]
save_model = True if train else False

new_model = CNN(folder=folder, starting_models=starting_models, mode='p', modality=modality, train_indices=train_indices, test_indices=test_indices, save_model=save_model, labels=labels)