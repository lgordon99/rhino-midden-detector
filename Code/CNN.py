'''
cnn by Lucia Gordon
'''

# imports
import random
import torch
import torch.nn as nn
import numpy as np
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
    def __init__(self, mode, modality, query, train_indices, test_indices):
        # global variables
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.accuracy = 0
        self.acc_empty = 0
        self.acc_middens = 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0
        self.accuracies = []
        self.results = []
        self.fraction_middens_found = []

        # labels
        labels = np.load('data/labels.npy')
        train_labels = np.take(labels, train_indices)
        test_labels = np.take(labels, test_indices)

        # identifiers
        identifiers = np.load('data/identifiers.npy')
        train_identifiers = np.take(identifiers, train_indices)
        test_identifiers = np.take(identifiers, test_indices)

        # maximum pixel values
        max_pixel_vals = np.load('data/max-pixel-vals.npy')
        train_max_pixel_vals = np.take(max_pixel_vals, train_indices)

        images = []
        train_images = []
        test_images = []
        models = []

        if len(modality) < 4: # if there is no fusing
            if 't' in modality:
                images.append(list(np.load('data/thermal/thermal-images.npy')))
            if 'r' in modality:
                images.append(list(np.load('data/rgb/rgb-images.npy')))
            if 'l' in modality:
                images.append(list(np.load('data/lidar/lidar-images.npy')))
        else: # fusing
            modality = modality.split('-')[0] # remove the "fused" part of the string
            
            if modality == 'tr':
                images.append(list(np.load('data/fused/tr-fused.npy')))
            elif modality == 'tl':
                images.append(list(np.load('data/fused/tl-fused.npy')))
            elif modality == 'rl':
                images.append(list(np.load('data/fused/rl-fused.npy')))
            elif modality == 'trl':
                images.append(list(np.load('data/fused/trl-fused.npy')))

        images = custom_zip(np.array(images)) # switch from grouping by modality to grouping by image
        train_images = np.take(images, train_indices, axis=0)
        test_images = np.take(images, test_indices, axis=0)
        models = [self.initialize_model() for _ in range(len(images[0]))]
        test_loader = self.make_loader(test_images, test_labels, test_identifiers, len(test_images))
        self.test(models, test_loader, weights=1/len(models)*np.ones(len(models)))
        self.accuracies.append(self.accuracy)

        # train & test models
        if mode == 'p':
            train_loader = self.make_loader(train_images, train_labels, train_identifiers, batch_size=10)
            self.passive_train(models, train_loader, epochs=10)
            self.test(models, test_loader, weights=1/len(models)*np.ones(len(models)))
        elif mode == 'a':
            self.active_train(models, train_images, train_labels, train_identifiers, train_max_pixel_vals, test_loader, batch_size=10, labeling_budget=800, query=query)

    def initialize_model(self):
        model = vgg16(weights='VGG16_Weights.DEFAULT').to(self.device) # imports a pretrained vgg16 CNN

        for parameter in model.parameters(): # freeze all parameters
            parameter.requires_grad = False
        
        model.classifier = torch.nn.Sequential(torch.nn.Linear(in_features=25088, out_features=4096),
                                            torch.nn.ReLU(inplace=True),
                                            torch.nn.Dropout(),
                                            torch.nn.Linear(in_features=4096, out_features=4096),
                                            torch.nn.ReLU(inplace=True),
                                            torch.nn.Dropout(),
                                            torch.nn.Linear(in_features=4096, out_features=1),
                                            torch.nn.Sigmoid()) # unfreeze classifier parameters
        
        print('Using ' + str(torch.cuda.device_count()) + ' GPU(s)')
        model.eval()

        return model

    def transform_images(self, images):
        transformed_images = torch.empty((images.shape[0],images.shape[1],images.shape[4],images.shape[2],images.shape[3])) # (batch_size, # modalities, 3, 224, 224)
        
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
        for i in range(len(data)+1): 
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
    
    def passive_train(self, models, train_loader, epochs):
        for i in range(len(models)): # for each model
            models[i].train()
            criterion = torch.nn.BCELoss()
            optimizer = torch.optim.Adam(models[i].parameters(), lr=0.0001)

            for epoch in range(epochs):  # loop over the dataset multiple times
                total_loss = 0

                for j in range(len(train_loader)): # for each batch
                    images = torch.FloatTensor(np.array(list(map(list, zip(*train_loader[j][0])))[i])).to(self.device) # extract the images in modality i
                    # images = torch.FloatTensor(np.array(list(map(list, zip(*train_loader[j][0].numpy())))[i])).to(self.device) # extract the images in modality i
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
                            
            models[i].eval()

    def query_indices(self, query, models, images, max_pixel_vals, unused_indices, batch_size, images_labeled, weights): # selects indices for querying for the user
        if query == 'r': # random
            queried_indices = random.sample(unused_indices, batch_size) # pick indices randomly

        elif query == 'u' or query == 'c': # uncertainty or positive certainty
            with torch.no_grad(): # no gradients are calculated for model inference
                scores = np.zeros(len(unused_indices)) # one score per image

                for i in range(len(models)): # for each modality
                    indices = [] # indices for inference
                    outputs = [] # model outputs

                    for j in range(len(unused_indices)+1): # for each of the unused indices
                        if (j % 3000 == 0 and j != 0) or (j == len(unused_indices)): # forming batches of max 3,000 to limit computational load
                            inference_images = torch.FloatTensor(np.array(list(map(list, zip(*self.transform_images(np.take(images, indices, axis=0)))))[i])).to(self.device) # extract the images in modality j
                            outputs += list(nn.DataParallel(models[i].to(self.device))(inference_images).flatten().cpu().detach().numpy()) # get the model's predictions using multiple GPUs
                            indices = [] # empty the list of indices for a new batch
                        
                        if j != len(unused_indices): # while there are still more indices left to be predicted on
                            indices.append(unused_indices[j]) # add the index to the list for inference
                    
                    scores += weights[i] * np.array(outputs) # update the image's score based on the model's output

            if query == 'u': # uncertainty
                queried_indices = list(np.take(unused_indices, np.argsort(abs(scores-0.5))[:batch_size]))
            
            elif query == 'c': # positive certainty
                queried_indices = list(np.take(unused_indices, np.flip(np.argsort(scores))[:batch_size]))            
        
        elif query == 'd': # disagree
            with torch.no_grad(): # no gradients are calculated for model inference
                indices = []
                differences = []

                for i in range(len(unused_indices)+1):
                    if (i % 3000 == 0 and i != 0) or (i == len(unused_indices)):
                        inference_images_1 = torch.FloatTensor(np.array(list(map(list, zip(*self.transform_images(np.take(images, indices, axis=0)))))[0])).to(self.device) # extract the images in modality j
                        inference_images_2 = torch.FloatTensor(np.array(list(map(list, zip(*self.transform_images(np.take(images, indices, axis=0)))))[1])).to(self.device) # extract the images in modality j
                        outputs_1 = nn.DataParallel(models[0].to(self.device))(inference_images_1).flatten().cpu().detach().numpy()
                        outputs_2 = nn.DataParallel(models[1].to(self.device))(inference_images_2).flatten().cpu().detach().numpy()
                        differences += list(abs(outputs_1-outputs_2))
                        indices = []

                    if i != len(unused_indices):
                        indices.append(unused_indices[i])
                
                queried_indices = list(np.take(unused_indices, np.flip(np.argsort(differences))[:batch_size])) # selects the images with the biggest difference in prediction from the two models

        elif query == 'v' or query == 'm' or query == 'a': # rule violation or multimodAL or ablation
            # if ((query == 'v' or query == 'm') and images_labeled == 0) or query == 'a': # jump-start m and v
            if query == 'a': # no jump start
                queried_indices = list(np.take(unused_indices, np.flip(np.argsort(np.take(max_pixel_vals, unused_indices)))[:batch_size]))
            
            # if (query == 'v' or query == 'm') and images_labeled != 0: # multimodAL after first round
            if query == 'v' or query == 'm': # no jump start
                i = 0
                queried_indices = [] # initialize an array to store the indices for querying

                while len(queried_indices) < batch_size: # while the number of indices selected for querying is less than the batch size
                    index = unused_indices[np.flip(np.argsort(np.take(max_pixel_vals, unused_indices)))[i]] # sort the indices according to which ones correspond to images with the highest maximum pixel values and select the ith one

                    with torch.no_grad(): # no gradients should be computed during model inference
                        score = 0 # the image starts with a score of 0

                        for j in range(len(models)): # for each modality
                            image = torch.FloatTensor(np.array(list(map(list, zip(*self.transform_images(np.array([images[i]])))))[j])).to(self.device) # extract the image in modality j
                            output = models[j].to(self.device)(image).flatten().cpu().detach().numpy() # model's probability of the image being a midden
                            score += weights[j] * output # increment the score          
                    
                    # predicted = np.around(score)
                    predicted = np.random.binomial(1, score) # assign the image to a class using a binomial draw to reflect the model(s)' uncertainty
                    i += 1 # move on to the next image

                    if query == 'm' and predicted[0] == 1: # if the image is predicted to be a midden
                        queried_indices += [index] # add it to the list for querying
                    
                    if query == 'v' and predicted[0] == 0: # if the image is predicted to be empty
                        queried_indices += [index] # add it to the list for querying
                    
                    if i == len(unused_indices): # if all the images have been predicted on
                        break # exit the loop

        return queried_indices # send the queried indices back for training
    
    def active_train(self, models, images, labels, identifiers, max_pixel_vals, test_loader, batch_size, labeling_budget, query):
        print('Query =', query)
        used_indices = [] # initially no images have been used
        unused_indices = list(np.arange(len(images))) # each image in the unlabeled pool gets an index
        images_labeled = 0 # no images start out labeled
        total_middens = np.sum(labels) # total number of middens in the training pool
        correct = np.zeros(len(models)) # vector containing how many images each model has classified correctly, which is 0 at the start
        weights = 1/len(models)*np.ones(len(models)) # the outputs of each model are weighted equally at the start
        
        while images_labeled < labeling_budget: # while more labels can still be provided
            queried_indices = self.query_indices(query, models, images, max_pixel_vals, unused_indices, batch_size, images_labeled, weights) # select images to query
            used_indices += queried_indices # add the queried indices to the list of used indices

            if len(models) > 1: # if there are multiple modalities
                with torch.no_grad(): # no gradients should be computed for model inference
                    for i in range(len(models)): # for each modality
                        queried_images = torch.FloatTensor(np.array(list(map(list, zip(*np.take(self.transform_images(images), queried_indices, axis=0))))[i])).to(self.device) # extract the images in modality j
                        queried_labels = torch.tensor(np.take(labels, queried_indices)).to(torch.float64) # the user provides labels for the queried images
                        outputs = models[i].to(self.device)(queried_images).flatten().cpu().detach().to(torch.float64) # the model assigns each image a value between 0 and 1, which is its probability of containing a midden
                        predicted = np.around(outputs) # round the outputs to get the predicted class for each image
                        correct[i] += (predicted == queried_labels).sum().item() # count how many images the model classified correctly

                weights = np.array(correct)/np.sum(correct) # update the weights according to the new classifications         
                print('Weights =', weights)

            for i in reversed(range(len(unused_indices))): # for each unused index
                if unused_indices[i] in used_indices: # if the index has been used
                    del unused_indices[i] # remove the index from the unused list
            
            print('Labels left =', labeling_budget-images_labeled)
            images_labeled = len(used_indices) # the number of images labeled is however many indices have been used
            train_indices = used_indices.copy() # initialize a list of indices for training

            if np.sum(np.take(labels, train_indices)) > 0: # if at least one midden has been found
                while np.sum(np.take(labels, train_indices)) < len(train_indices)/2: # while less than half the indices in the training list correspond to middens
                    del train_indices[random.choice(np.where(np.take(labels, train_indices)==0)[0])] # randomly remove an index corresponding to an empty image
                    
            train_images = np.take(images, train_indices, axis=0) # extract the images for training
            train_labels = np.take(labels, train_indices) # extract the labels for training
            train_identifiers = np.take(identifiers, train_indices) # extract the identifiers for training
            train_loader = self.make_loader(train_images, train_labels, train_identifiers, batch_size) # generate the training loader
            models = [self.initialize_model() for _ in range(len(images[0]))] # reset the model(s)
            self.passive_train(models, train_loader, epochs=10) # train the model(s)
            self.test(models, test_loader, weights) # test the model(s)
            self.accuracies.append(self.accuracy) # save the accuracy
            self.fraction_middens_found.append(np.sum(np.take(labels, used_indices))/total_middens) # save the fraction of the middens found
            print('Total middens found =', np.sum(np.take(labels, used_indices)))
            print('Fraction middens found =', round(np.sum(np.take(labels, used_indices))/total_middens,3))

    def test(self, models, test_loader, weights):
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
                # print('length scores:', len(scores))

                for j in range(len(test_loader[i][0][0])): # for each modality
                    images = torch.FloatTensor(np.array(list(map(list, zip(*test_loader[i][0])))[j])).to(self.device) # extract the images in modality j
                    # images = torch.FloatTensor(np.array(list(map(list, zip(*test_loader[i][0].numpy())))[j])).to(self.device) # extract the images in modality j
                    # print('length batch test images:', len(images))
                    labels = torch.tensor(test_loader[i][1]).to(torch.float64)
                    # print('length batch test labels:', len(labels))
                    outputs = models[j].to(self.device)(images).flatten().cpu().detach().to(torch.float64)
                    scores += weights[j] * outputs                

                # print('scores =', scores)
                predicted = np.around(scores)
                # print('predicted =', predicted)
                total += len(predicted) # number of images in the batch
                correct += (predicted == labels).sum().item() # number of images classified correctly
                indices_empty = np.where(labels == 0)[0] # indices of no midden images
                total_empty += len(indices_empty) # number of images with no middens
                correct_empty += (np.take(predicted, indices_empty) == np.take(labels, indices_empty)).sum().item() # number of true negatives
                indices_middens = np.where(labels == 1)[0] # indices of midden images
                total_midden += len(indices_middens) # number of images with middens
                correct_middens += (np.take(predicted, indices_middens) == np.take(labels, indices_middens)).sum().item() # number of true positives
                predicted_middens += predicted.sum().item() # true positives + false positives

        self.accuracy = round(correct/total, 3)
        print(f'Accuracy of the neural network on the {total} test images = {self.accuracy}')
        self.acc_empty = round(correct_empty/total_empty, 3)
        print(f'Accuracy on images without middens = {self.acc_empty}')
        self.acc_middens = round(correct_middens/total_midden, 3)
        print(f'Accuracy on images with middens = {self.acc_middens}')

        if predicted_middens > 0:
            self.precision = round(correct_middens/predicted_middens,3)
        else:
            self.precision = 0

        print(f'Precision = {self.precision}') # fraction of images classified as having middens that actually have middens

        self.recall = round(correct_middens/total_midden,3)
        print(f'Recall = {self.recall}') # fraction of images with middens classified as having middens
        
        if self.precision == 0 and self.recall == 0:
            self.f1 = 0
        else:
            self.f1 = round(2*self.precision*self.recall/(self.precision+self.recall), 3)
        
        print(f'F1 score = {self.f1}') # harmonic mean of precision and recall
        self.results = [self.accuracy, self.acc_empty, self.acc_middens, self.precision, self.recall, self.f1]