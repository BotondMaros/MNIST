
from __future__ import print_function
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle as pkl
import time
import copy
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F 
from torchvision import datasets, transforms, models
from torch.autograd import Variable

#todo -> pickle to tensor, transformations, bounding box selection, image classifictation, tuning
# pretrained models? 

data_folder = './data/'
train_filename = data_folder + 'train_images.pkl'
labels_filename = data_folder + 'train_labels.csv'
test_filename = data_folder + 'test_images.pkl'
validation_cutoff = 35000
num_classes = 10
batch_size = 16
num_epochs = 20

#resnet = models.ResNet()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_pickle_data(filename):
    pickle_off = open(filename,'rb')
    data = pkl.load(pickle_off)
    return data

def get_labels_from_csv():
    return pd.read_csv(labels_filename)


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

images = get_pickle_data(train_filename)
labels = get_labels_from_csv()

print(labels)

#print(image.shape)
#print(device)
#plt.imshow(image)
#plt.show()

