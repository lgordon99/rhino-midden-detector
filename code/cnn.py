'''cnn.py by Lucia Gordon'''

# imports
import geopandas as gpd
import numpy as np
import os
import random
import shutil
import torch
import torch.nn as nn
import utils
from send_receive_emails import SendReceiveEmail
from shapely.geometry import Polygon
from torchvision.models import vgg16
from torchvision.transforms import transforms
from torchsummary import summary

# global functions
def custom_zip(array):
    zipped_array = []

    for i in range(array.shape[1]):
        sub_array = []

        for j in range(array.shape[0]):
            sub_array.append(array[j][i])
        
        zipped_array.append(sub_array)
    
    return np.array(zipped_array)

# CNN
class CNN:
    def __init__(self, starting_models, mode, modality, query, train_indices, test_indices, inference_indices, save_models):
        constants = utils.get_constants()
        
        # global variables
        self.project_dir = utils.get_project_dir()
        self.site = utils.get_site()
        self.modality = modality
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = [] # [accuracy, precision, recall, f1, (fraction_middens_found)]
        self.starting_models = starting_models
        self.save_models = save_models
        self.THERMAL_INTERVAL = int(constants[0][1])
        self.THERMAL_STRIDE = int(constants[1][1])
        self.THERMAL_LEFT = float(constants[2][1])
        self.THERMAL_TOP = float(constants[3][1])
        self.THERMAL_PIXEL_WIDTH = float(constants[4][1])
        self.THERMAL_PIXEL_HEIGHT = float(constants[5][1])
        self.NUM_HORIZONTAL = int(constants[8][1])

        train = True if train_indices is not None else False
        test = True if test_indices is not None else False
        inference = True if inference_indices is not None else False

        # identifiers
        identifiers = np.load(f'{self.project_dir}/{self.site}/data/identifiers.npy')
        if train: train_identifiers = np.take(identifiers, train_indices)
        if test: test_identifiers = np.take(identifiers, test_indices)

        # maximum pixel values
        max_pixel_vals = np.load(f'{self.project_dir}/{self.site}/data/max-pixel-vals.npy')
        if train: train_max_pixel_vals = np.take(max_pixel_vals, train_indices)

        images = []
        if train: train_images = []
        if test: test_images = []

        if len(modality) < 4: # no fusing
            if 't' in modality:
                images.append(list(np.load(f'{self.project_dir}/{self.site}/data/thermal/thermal-images.npy')))
            if 'r' in modality:
                images.append(list(np.load(f'{self.project_dir}/{self.site}/data/rgb/rgb-images.npy')))
            if 'l' in modality:
                images.append(list(np.load(f'{self.project_dir}/{self.site}/data/lidar/lidar-images.npy')))
        
        else: # fusing
            modality = modality.split('-')[0] # remove the "fused" part of the string
            
            if modality == 'tr':
                images.append(list(np.load(f'{self.project_dir}/{self.site}/data/fused/tr-fused.npy')))
            elif modality == 'tl':
                images.append(list(np.load(f'{self.project_dir}/{self.site}/data/fused/tl-fused.npy')))
            elif modality == 'rl':
                images.append(list(np.load(f'{self.project_dir}/{self.site}/data/fused/rl-fused.npy')))
            elif modality == 'trl':
                images.append(list(np.load(f'{self.project_dir}/{self.site}/data/fused/trl-fused.npy')))

        images = custom_zip(np.array(images)) # switch from grouping by modality to grouping by image
        self.num_models = images.shape[1]

        if train: train_images = np.take(images, train_indices, axis=0)
        if test: test_images = np.take(images, test_indices, axis=0)
        
        models = self.initialize_model(self.starting_models, self.num_models)
        test_loader = self.make_loader(test_images, test_labels, test_identifiers, batch_size=10) if test else None
        # if inference: self.inference(models, self.make_inference_loader(images, batch_size=10), weights=1/(0.947 + 0.65 + 0.655) * np.array([0.947, 0.65, 0.655]), path=f'{self.project_dir}/{self.site}/data/scores')
        if inference: self.inference(models, self.make_inference_loader(images, batch_size=10), weights=np.array(2*[0.5]), path=f'{self.project_dir}/{self.site}/data/scores')

        if train:
            if os.path.exists(f'{self.project_dir}/{self.site}/data/labels.npy'):
                labels = np.load(f'{self.project_dir}/{self.site}/data/labels.npy')
                train_labels = np.take(labels, train_indices)
                if test: test_labels = np.take(labels, test_indices)

            if mode == 'p': # passive
                train_loader = self.make_loader(train_images, train_labels, train_identifiers, batch_size=10)
                self.passive_train(models, train_loader, epochs=10)
                if test: self.test(models, test_loader, weights=1/len(models) * np.ones(len(models)))
            
            elif mode == 'a': # active
                if test: self.test(models, test_loader, weights=1/len(models) * np.ones(len(models)), fraction_middens_found=0)
                self.active_train(models, train_images, None, train_identifiers, train_max_pixel_vals, test_loader, batch_size=10, labeling_budget=500, query=query)
    
    def initialize_model(self, starting_models, num_models):
        print(f'Using {torch.cuda.device_count()} GPU(s)')

        if starting_models[0] == 'vgg16':
            model = vgg16(weights = 'VGG16_Weights.DEFAULT').to(self.device) # imports a pretrained vgg16 CNN

            for parameter in model.parameters(): # freeze all parameters
                parameter.requires_grad = False
            
            model.classifier = torch.nn.Sequential(torch.nn.Linear(in_features=25088, out_features=4096),
                                                   torch.nn.ReLU(inplace=True),
                                                   torch.nn.Dropout(),
                                                   torch.nn.Linear(in_features=4096, out_features=4096),
                                                   torch.nn.ReLU(inplace = True),
                                                   torch.nn.Dropout(),
                                                   torch.nn.Linear(in_features=4096, out_features=1),
                                                   torch.nn.Sigmoid()) # unfreeze classifier parameters
            model.eval()
            return num_models * [model]
        else:
            print(starting_models)
            return [torch.load(starting_models[i]).eval() for i in range(len(starting_models))]

    def transform_images(self, images):
        transformed_images = torch.empty((images.shape[0], images.shape[1], images.shape[4], images.shape[2], images.shape[3])) # (batch_size, # modalities, 3, 224, 224)
        
        for i in range(images.shape[0]):
            for j in range(images.shape[1]):
                transformed_images[i][j] = self.transform(images[i][j])

        return transformed_images.numpy()

    def make_loader(self, images, labels, identifiers, batch_size):
        data = list(map(list, zip(images, labels, identifiers))) # each image gets grouped with its label and identifier
        data = random.sample(data, len(data)) # shuffle the training data
        loader = []
        image_batch = []
        label_batch = []
        identifier_batch = []

        # batch the data
        for i in range(len(data) + 1): 
            if (i % batch_size == 0 and i != 0) or (i == len(data)):
                loader.append([self.transform_images(np.array(image_batch)), label_batch, identifier_batch])
                image_batch = []
                label_batch = []
                identifier_batch = []

            if i != len(data):
                image_batch.append(data[i][0])
                label_batch.append(data[i][1])
                identifier_batch.append(data[i][2])
        
        return loader

    def make_inference_loader(self, images, batch_size):
        data = images
        loader = []
        image_batch = []

        # batch the data
        for i in range(len(data) + 1): 
            if (i % batch_size == 0 and i != 0) or (i == len(data)):
                loader.append([self.transform_images(np.array(image_batch))])
                image_batch = []

            if i != len(data):
                image_batch.append(data[i])
        
        print('made inference loader')
        return loader
    
    def passive_train(self, models, train_loader, epochs):
        for i in range(len(models)): # for each model
            models[i].train()
            criterion = torch.nn.BCELoss()
            optimizer = torch.optim.Adam(models[i].parameters(), lr = 0.0001)

            for epoch in range(epochs):  # loop over the dataset multiple times
                total_loss = 0

                for j in range(len(train_loader)): # for each batch
                    images = torch.FloatTensor(np.array(list(map(list, zip(*train_loader[j][0])))[i])).to(self.device) # extract the images in modality i
                    # print('length batch training images:', len(images))
                    labels = torch.tensor(train_loader[j][1]).to(torch.float64).to(self.device)
                    # print('length batch training labels:', len(labels))
                    optimizer.zero_grad() # zero the parameter gradients
                    outputs = models[i].to(self.device)(images).flatten().to(torch.float64) # forward pass
                    loss = criterion(outputs, labels)
                    loss.backward() # backward pass
                    optimizer.step() # optimization
                    total_loss += loss.item()

                print('Epoch ' + str(epoch+1) + ' loss = ' + str(round(total_loss,3)))

            if self.save_models:
                os.makedirs(f'{self.project_dir}/{self.site}/models', exist_ok=True)
                torch.save(models[i], f'{self.project_dir}/{self.site}/models/model-{self.modality}')
                print(f'{self.modality} model saved')

            models[i].eval()

    def get_image_center_pixels(self, identifier):
        row = np.floor(identifier / self.NUM_HORIZONTAL)
        col = identifier - self.NUM_HORIZONTAL * np.floor(identifier / self.NUM_HORIZONTAL)
        x_pixels = col * (self.THERMAL_STRIDE + self.THERMAL_INTERVAL / 2) + self.THERMAL_INTERVAL / 2
        y_pixels = row * (self.THERMAL_STRIDE + self.THERMAL_INTERVAL / 2) + self.THERMAL_INTERVAL / 2

        return x_pixels, y_pixels

    def get_image_center_meters(self, x_pixels, y_pixels):
        x = self.THERMAL_LEFT + x_pixels * self.THERMAL_PIXEL_WIDTH
        y = self.THERMAL_TOP + y_pixels * self.THERMAL_PIXEL_HEIGHT

        return x, y
    
    def make_shapefile(self, identifiers, batch_number):
        empty_list = len(identifiers) * [''] # placeholders for labels
        centers_in_meters = []

        for identifier in identifiers:
            x_pixels, y_pixels = self.get_image_center_pixels(identifier)
            x, y = self.get_image_center_meters(x_pixels, y_pixels)
            centers_in_meters.append([x, y])

        polygons = [Polygon([[center[0] - 10, center[1] - 10], [center[0] + 10, center[1] - 10], [center[0] + 10, center[1] + 10], [center[0] - 10, center[1] + 10]]) for center in centers_in_meters] # each image is 20 x 20 m
        gdf = gpd.GeoDataFrame(geometry = polygons)
        gdf['id'] = identifiers
        gdf['label'] = empty_list
        if os.path.exists(f'{self.project_dir}/{self.site}/data/shapefile/'): shutil.rmtree(f'{self.project_dir}/{self.site}/data/shapefile/')
        os.mkdir(f'{self.project_dir}/{self.site}/data/shapefile/')
        gdf.to_file(f'{self.project_dir}/{self.site}/data/shapefile/batch-{batch_number}-shapefile.shp')
        shutil.make_archive(f'{self.project_dir}/{self.site}/data/batch-{batch_number}-shapefile', 'zip', f'{self.project_dir}/{self.site}/data/shapefile')

    def query_indices(self, query, models, images, max_pixel_vals, unlabeled_indices, batch_size, weights): # selects indices for querying for the user
        if query == 'r': # random
            queried_indices = random.sample(unlabeled_indices, batch_size) # pick indices randomly

        elif query == 'u' or query == 'c': # uncertainty or positive certainty
            with torch.no_grad(): # no gradients are calculated for model inference
                scores = np.zeros(len(unlabeled_indices)) # one score per image

                for i in range(len(models)): # for each modality
                    indices = [] # indices for inference
                    outputs = [] # model outputs

                    for j in range(len(unlabeled_indices) + 1): # for each of the unused indices
                        if (j % 3000 == 0 and j != 0) or (j == len(unlabeled_indices)): # forming batches of max 3,000 to limit computational load
                            inference_images = torch.FloatTensor(np.array(list(map(list, zip(*self.transform_images(np.take(images, indices, axis = 0)))))[i])).to(self.device) # extract the images in modality j
                            outputs += list(nn.DataParallel(models[i].to(self.device))(inference_images).flatten().cpu().detach().numpy()) # get the model's predictions using multiple GPUs
                            indices = [] # empty the list of indices for a new batch
                        
                        if j != len(unlabeled_indices): # while there are still more indices left to be predicted on
                            indices.append(unlabeled_indices[j]) # add the index to the list for inference
                    
                    scores += weights[i] * np.array(outputs) # update the image's score based on the model's output

            if query == 'u': # uncertainty
                queried_indices = list(np.take(unlabeled_indices, np.argsort(abs(scores - 0.5))[:batch_size]))
            
            elif query == 'c': # positive certainty
                queried_indices = list(np.take(unlabeled_indices, np.flip(np.argsort(scores))[:batch_size]))            
        
        elif query == 'd': # disagree
            with torch.no_grad(): # no gradients are calculated for model inference
                indices = []
                differences = []

                for i in range(len(unlabeled_indices)+1):
                    if (i % 3000 == 0 and i != 0) or (i == len(unlabeled_indices)):
                        inference_images_1 = torch.FloatTensor(np.array(list(map(list, zip(*self.transform_images(np.take(images, indices, axis = 0)))))[0])).to(self.device) # extract the images in modality j
                        inference_images_2 = torch.FloatTensor(np.array(list(map(list, zip(*self.transform_images(np.take(images, indices, axis = 0)))))[1])).to(self.device) # extract the images in modality j
                        outputs_1 = nn.DataParallel(models[0].to(self.device))(inference_images_1).flatten().cpu().detach().numpy()
                        outputs_2 = nn.DataParallel(models[1].to(self.device))(inference_images_2).flatten().cpu().detach().numpy()
                        differences += list(abs(outputs_1 - outputs_2))
                        indices = []

                    if i != len(unlabeled_indices):
                        indices.append(unlabeled_indices[i])
                
                queried_indices = list(np.take(unlabeled_indices, np.flip(np.argsort(differences))[:batch_size])) # selects the images with the biggest difference in prediction from the two models

        elif query == 'a': # ablation
            queried_indices = list(np.take(unlabeled_indices, np.flip(np.argsort(np.take(max_pixel_vals, unlabeled_indices)))[:batch_size]))

        elif query == 'v' or query == 'm': # rule violation or multimodAL
            i = 0
            queried_indices = [] # initialize an array to store the indices for querying

            while len(queried_indices) < batch_size: # while the number of indices selected for querying is less than the batch size
                index = unlabeled_indices[np.flip(np.argsort(np.take(max_pixel_vals, unlabeled_indices)))[i]] # sort the indices according to which ones correspond to images with the highest maximum pixel values and select the ith one

                with torch.no_grad(): # no gradients should be computed during model inference
                    score = 0 # the image starts with a score of 0

                    for j in range(len(models)): # for each modality
                        image = torch.FloatTensor(np.array(list(map(list, zip(*self.transform_images(np.array([images[i]])))))[j])).to(self.device) # extract the image in modality j
                        output = models[j].to(self.device)(image).flatten().cpu().detach().numpy() # model's probability of the image being a midden
                        score += weights[j] * output # increment the score          
                
                predicted = np.random.binomial(1, score) # assign the image to a class using a binomial draw to reflect the model(s)' uncertainty
                i += 1 # move on to the next image

                if query == 'm' and predicted[0] == 1: # if the image is predicted to be a midden
                    queried_indices += [index] # add it to the list for querying
                
                if query == 'v' and predicted[0] == 0: # if the image is predicted to be empty
                    queried_indices += [index] # add it to the list for querying
                
                if i == len(unlabeled_indices): # if all the images have been predicted on
                    break # exit the loop

        return queried_indices # send the queried indices back for training
    
    def active_train(self, models, images, labels, identifiers, max_pixel_vals, test_loader, batch_size, labeling_budget, query):
        print(f'Query = {query}')
        batch_number = 0
        unlabeled_indices = np.arange(len(images)) # each image in the unlabeled pool gets an index
        labeled_identifiers_path = f'{self.project_dir}/{self.site}/data/labeled-ids.npy'

        if os.path.exists(labeled_identifiers_path): # if some labels are provided
            labeled_identifiers = np.load(f'{self.project_dir}/{self.site}/data/labeled-ids.npy') # get labeled identifiers
            print(f'Labeled identifiers =\n{labeled_identifiers}')
            batch_number = int(len(labeled_identifiers)/10)
            labeled_indices = list(np.where(np.isin(identifiers, labeled_identifiers.T[0]))[0]) # get indices corresponding to labeled identifiers
            unlabeled_indices = np.delete(unlabeled_indices, np.where(np.isin(unlabeled_indices, labeled_indices))[0]) # set the rest of the indices as unlabeled
            total_middens = np.sum(labeled_identifiers.T[1]) # total number of middens in the training pool

            # train model on existing labeled data
            train_indices = labeled_indices.copy() # initialize a list of indices for training
            print(f'Train identifiers = {identifiers[train_indices]}')
            train_labels = [labeled_identifiers.T[1][np.where(labeled_identifiers.T[0] == identifiers[index])[0][0]] for index in train_indices]
            print(f'Train labels = {train_labels}')

            # balance training data if possible
            if np.sum(train_labels) > 0: # if at least one midden has been found
                while np.sum(train_labels) < len(train_indices) / 2: # while less than half the indices in the training list correspond to middens
                    train_indices = np.delete(train_indices, random.choice(np.where(np.array(train_labels) == 0)[0])) # randomly delete an index corresponding to an empty image
                    train_labels = [labeled_identifiers.T[1][np.where(labeled_identifiers.T[0] == identifiers[index])[0][0]] for index in train_indices]
            
            print(f'Train labels after balancing = {train_labels}')

            train_images = np.take(images, train_indices, axis=0) # extract the images for training
            train_identifiers = identifiers[train_indices] # extract the identifiers for training
            train_loader = self.make_loader(train_images, train_labels, train_identifiers, batch_size) # generate the training loader
            self.passive_train(models, train_loader, epochs=10) # train the model(s)
        else:
            labeled_indices = [] # if no labels are provided or all labels are provided and AL is being simulated
            labeled_identifiers = [] if labels is None else list(map(list, zip(identifiers, labels)))

        print(f'Unlabeled indices length = {len(unlabeled_indices)}')
        correct = np.zeros(len(models)) # vector containing how many images each model has classified correctly, which is 0 at the start
        weights = 1 / len(models) * np.ones(len(models)) # the outputs of each model are weighted equally at the start
        
        # active learning loop
        while len(labeled_indices) < labeling_budget: # while more labels can still be provided
            print(f'Labels left = {labeling_budget - len(labeled_indices)}')
            batch_number += 1
            queried_indices = self.query_indices(query, models, images, max_pixel_vals, unlabeled_indices, batch_size, weights) # select images to query
            queried_identifiers = identifiers[queried_indices]
            labeled_indices += queried_indices # add the queried indices to the list of used indices
            unlabeled_indices = np.delete(unlabeled_indices, np.where(np.isin(unlabeled_indices, queried_indices))[0]) # set the rest of the indices as unlabeled
            print(f'Queried identifiers = {queried_identifiers}')                

            # get batch labels
            if labels is None: # if none or some labels are provided
                self.make_shapefile(queried_identifiers, batch_number) # generate shapefile to be emailed
                id_label_dict = SendReceiveEmail(folder=f'{self.project_dir}/{self.site}', batch_number=batch_number).batch_labels # the user provides labels for the queried images            
                print(f'id label dict = {id_label_dict}')
                queried_labeled_identifiers = np.array([[identifier, id_label_dict[identifier]] for identifier in queried_identifiers]) # turn the dict to array and ensure labels are in the right order
                queried_labels = queried_labeled_identifiers.T[1]
                labeled_identifiers = np.array(list(labeled_identifiers) + list(queried_labeled_identifiers))
                os.remove(f'{self.project_dir}/{self.site}/data/batch-{batch_number}-shapefile.zip')
                np.save(labeled_identifiers_path, labeled_identifiers)
                
                print(f'Queried labeled identifiers =\n{queried_labeled_identifiers}')
                print(f'Labeled identifiers = {labeled_identifiers}')
            else: # all labels are provided and AL is being simulated
                queried_labels = labels[queried_indices]
            
            print(f'Queried labels = {queried_labels}')

            # update relative model weights
            if len(models) > 1: # if there are multiple modalities
                with torch.no_grad(): # no gradients should be computed for model inference
                    for i in range(len(models)): # for each modality
                        queried_images = torch.FloatTensor(np.array(list(map(list, zip(*np.take(self.transform_images(images), queried_indices, axis=0))))[i])).to(self.device) # extract the images in modality j
                        outputs = models[i].to(self.device)(queried_images).flatten().cpu().detach().to(torch.float64) # the model assigns each image a value between 0 and 1, which is its probability of containing a midden
                        predicted = np.around(outputs) # round the outputs to get the predicted class for each image
                        correct[i] += (predicted == torch.tensor(queried_labels).to(torch.float64)).sum().item() # count how many images the model classified correctly

                weights = np.array(correct) / np.sum(correct) # update the weights according to the new classifications         
                
                print(f'Weights = {np.around(weights, 3)}')
            
            # generate training data
            train_indices = labeled_indices.copy() # initialize a list of indices for training
            train_labels = [labeled_identifiers.T[1][np.where(labeled_identifiers.T[0] == identifiers[index])[0][0]] for index in train_indices]
            if labels is not None: print(f'labels[train_indices] = {labels[train_indices]}')

            # balance training data if possible
            if np.sum(train_labels) > 0: # if at least one midden has been found
                while np.sum(train_labels) < len(train_indices) / 2: # while less than half the indices in the training list correspond to middens
                    train_indices = np.delete(train_indices, random.choice(np.where(np.array(train_labels) == 0)[0])) # randomly delete an index corresponding to an empty image
                    train_labels = [labeled_identifiers.T[1][np.where(labeled_identifiers.T[0] == identifiers[index])[0][0]] for index in train_indices]
            
            print(f'Train labels = {train_labels}')

            train_images = np.take(images, train_indices, axis=0) # extract the images for training
            train_identifiers = identifiers[train_indices] # extract the identifiers for training
            train_loader = self.make_loader(train_images, train_labels, train_identifiers, batch_size) # generate the training loader
            models = self.initialize_model(self.starting_models, self.num_models) # reset the model(s)
            self.passive_train(models, train_loader, epochs=10) # train the model(s)
            
            if test_loader is not None:
                self.test(models, test_loader, weights, fraction_middens_found=np.sum(np.take(labels, labeled_indices)) / total_middens) # test the model(s)
                
                print('Total middens found =', np.sum(np.take(labels, labeled_indices)))
                print('Fraction middens found =', round(np.sum(np.take(labels, labeled_indices)) / total_middens, 3))

    def test(self, models, test_loader, weights, fraction_middens_found=None):
        correct = 0
        correct_empty = 0
        correct_middens = 0
        total = 0
        total_empty = 0
        total_midden = 0
        predicted_middens = 0

        with torch.no_grad(): # since we're not training, we don't need to calculate the gradients for our outputs
            for i in range(len(test_loader)): # for each batch
                scores = torch.zeros(len(test_loader[i][0])) # one score per image in the batch

                for j in range(len(test_loader[i][0][0])): # for each modality
                    images = torch.FloatTensor(np.array(list(map(list, zip(*test_loader[i][0])))[j])).to(self.device) # extract the images in modality j
                    labels = torch.tensor(test_loader[i][1]).to(torch.float64)
                    outputs = models[j].to(self.device)(images).flatten().cpu().detach().to(torch.float64)
                    scores += weights[j] * outputs                

                predicted = np.around(scores)
                total += len(predicted) # number of images in the batch
                correct += (predicted == labels).sum().item() # number of images classified correctly
                indices_empty = np.where(labels == 0)[0] # indices of no midden images
                total_empty += len(indices_empty) # number of images with no middens
                correct_empty += (np.take(predicted, indices_empty) == np.take(labels, indices_empty)).sum().item() # number of true negatives
                indices_middens = np.where(labels == 1)[0] # indices of midden images
                total_midden += len(indices_middens) # number of images with middens
                correct_middens += (np.take(predicted, indices_middens) == np.take(labels, indices_middens)).sum().item() # number of true positives
                predicted_middens += predicted.sum().item() # true positives + false positives

        accuracy = correct / total
        precision = correct_middens / predicted_middens if predicted_middens > 0 else 0
        recall = correct_middens / total_midden
        f1 = 0 if precision == 0 and recall == 0 else 2 * precision * recall / (precision + recall)
        self.results += [[accuracy, precision, recall, f1]] if fraction_middens_found is None else [[accuracy, precision, recall, f1, fraction_middens_found]]

        print(f'Accuracy of the neural network on the {total} test images = {round(accuracy, 3)}')
        print(f'Precision = {round(precision, 3)}') # fraction of images classified as having middens that actually have middens
        print(f'Recall = {round(recall, 3)}') # fraction of images with middens classified as having middens        
        print(f'F1 score = {round(f1, 3)}') # harmonic mean of precision and recall
    
    def inference(self, models, inference_loader, weights, path):
        print(weights)
        all_scores = []

        with torch.no_grad(): # since we're not training, we don't need to calculate the gradients for our outputs
            for i in range(len(inference_loader)): # for each batch
                scores = torch.zeros(len(inference_loader[i][0])) # one score per image in the batch

                for j in range(len(inference_loader[i][0][0])): # for each modality
                    images = torch.FloatTensor(np.array(list(map(list, zip(*inference_loader[i][0])))[j])).to(self.device) # extract the images in modality j
                    outputs = nn.DataParallel(models[j].to(self.device))(images).flatten().cpu().detach().to(torch.float64)
                    scores += weights[j] * outputs                

                all_scores += list(scores.numpy())
        
        np.save(path, all_scores)
        print('Scores saved')
