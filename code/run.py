'''run.py by Lucia Gordon'''

# imports
import argparse
import os
import random
import numpy as np
import utils
from cnn import CNN

# global variables
project_dir = utils.get_project_dir()
site = utils.get_site()

# process command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--trials', type=str, required=False, default='0') # range of integers inclusive ("0-29") or comma-separated list in ascending order ("30,33,37")
parser.add_argument('--train', type=utils.str_to_bool, required=False, default=False) # true ("True") or false ("False")
parser.add_argument('--test', type=utils.str_to_bool, required=False, default=False) # true ("True") or false ("False")
parser.add_argument('--mode', type=str, required=True) # passive ("p"), active ("a"), or inference ("i")
parser.add_argument('--modality', type=str, required=True) # thermal ("t"), RGB ("r"), LiDAR ("l"), thermal-RGB fused ("tr-fused"), thermal-LiDAR fused ("tl-fused"), RGB-LiDAR fused ("rl-fused"), and/or thermal-RGB-LiDAR fused ("trl-fused")
parser.add_argument('--starting_models', type=str, required=True) # VGG16 ("vgg16") or pre-trained ("PATHS TO MODELS")
parser.add_argument('--save_models', type=utils.str_to_bool, required=False, default=False) # true ("True") or false ("False")
parser.add_argument('--query', type=str, required=False, default='') # random ("r"), uncertainty ("u"), certainty ("c"), disagree ("d"), or multimodAL ("m")
parser.add_argument('--save_results', type=utils.str_to_bool, required=False, default=False) # true ("True") or false ("False")
args = parser.parse_args()

# process trial number(s)
if '-' in args.trials: # if a range of integers is provided
    start = int(args.trials.split('-')[0])
    stop = int(args.trials.split('-')[1])
    trials = np.arange(start, stop + 1)
elif ',' in args.trials: # if a list of integers is provided
    trials = [int(args.trials.split(',')[i]) for i in range(len(args.trials.split(',')))]
else: # if a single trial number is provided
    trials = [int(args.trials)]

starting_models = [args.starting_models.split(',')[i] for i in range(len(args.starting_models.split(',')))] if ',' in args.starting_models else [args.starting_models]
setting = f'{args.mode}-{args.modality}-{args.query}' # overall setting
identifiers = np.load(f'{project_dir}/{site}/data/identifiers.npy') # image identifiers

os.makedirs(f'{project_dir}/{site}/results/figs', exist_ok=True)

# functions
def train_test_split(labels): # select images for training and testing
    num_midden_imgs = np.sum(labels) # total number of images with middens
    test_indices = np.append(random.sample(list(np.where(labels == 1)[0]), round(0.2 * num_midden_imgs)), random.sample(list(np.where(labels == 0)[0]), round(0.2 * num_midden_imgs))) # select equal numbers of midden and empty images for testing
    train_indices = list(np.delete(list(range(len(labels))), test_indices)) # all the images not set aside for testing are for training

    return train_indices, test_indices

def downsample_empty_train_images(labels, train_indices, middens_empty_ratio_training):
    while np.sum(np.take(labels, train_indices)) < middens_empty_ratio_training * len(train_indices) / (middens_empty_ratio_training + 1): # if the fraction of middens in the training set is too low
        del train_indices[random.choice(np.where(np.take(labels, train_indices) == 0)[0])] # delete an empty image from the training set

    print(len(train_indices))
    print(np.sum(np.take(labels, train_indices)))
    print(np.sum(np.take(labels, test_indices)))
    print(len(test_indices))

    return train_indices

# run model
for trial in trials:
    print(f'Setting: {setting}')
    print(f'Trial {trial}')

    if args.test: # training and testing
        labels = np.load(f'{project_dir}/{site}/data/labels.npy') # image labels
        train_indices, test_indices = train_test_split(labels) # randomly split the dataset into training and testing
        inference_indices = None # no images are for inference

        if args.mode == 'p': # passive mode
            train_indices = downsample_empty_train_images(labels, train_indices, middens_empty_ratio_training=1) # downsample empty images in the training set

    elif args.train: # only training
        # train_indices = list(range(len(identifiers))) # all the images are for training
        train_indices = list(np.load(f'{project_dir}/{site}/data/labeled-indices.npy')) # indices for labeled images
        test_indices = None # no images are for testing
        inference_indices = None # no images are for inference
    
    elif args.mode == 'i': # only inference
        inference_indices = list(range(len(identifiers))) # all the images are for inference
        train_indices = None # no images are for training
        test_indices = None # no images are for testing

    model = CNN(starting_models, args.mode, args.modality, args.query, train_indices, test_indices, inference_indices, args.save_models) # run the model

    if args.save_results: # save results
        if os.path.exists(f'{project_dir}/{site}/results/{setting}.npy'): # check if the setting has a file
            results = np.load(f'{project_dir}/{site}/results/{setting}.npy', allow_pickle = True).tolist() # load the existing file
            print('loading in', results)
            if trial > len(results) - 1: # if the trial number is larger than the length of results
                results += [[] for i in range(trial + 1 - len(results))] # add empty lists to results
                
        else: # if the setting does not have a file
            results = [[] for i in range(trial + 1)] # initialize an empty results array

        results[trial] = model.results
        print('right before saving ', results)
        np.save(f'{project_dir}/{site}/results/{setting}', np.array(results, dtype = object))
