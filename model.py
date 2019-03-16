
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

def convert_images(images):
    convert = lambda image: (image*255).astype(np.uint8)
    converted = np.apply_along_axis(convert,0,images)
    return(converted)

def get_biggest_digit(image):
    blur = cv2.GaussianBlur(image,(7,7),0)
    _,img_bin = cv2.threshold(blur,127,255,cv2.THRESH_OTSU)
    contours,_ = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont = np.array(contours)
    largest = sorted(contours, key=lambda x: cv2.contourArea(x))[-2]
    index = contours.index(largest)
    white = (np.zeros(image.shape)).astype(np.uint8)
    mask = cv2.drawContours(white, contours,index,255,-1)
    return(mask)

def get_biggest_digit2(image):
    img_bin = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,5)
    contours,_ = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    largest = sorted(contours, key=lambda x: cv2.contourArea(x))[-2]
    index = contours.index(largest.all())
    white = (np.zeros(image.shape)).astype(np.uint8)
    mask = cv2.drawContours(white, contours,index,255,-1)
    return(mask)


def filter_images(images):
    fil = lambda image: get_biggest_digit(image)
    single = np.apply_along_axis(fil,0,images)
    return(single)

def filter_images2(images):
    for i in range(len(images)):
        images[i] = get_biggest_digit(images[i].copy())
    return(images)

images = get_pickle_data(train_filename)
labels = get_labels_from_csv()
'''
image = filter_images2(convert_images(images))[16]
plt.imshow(image)
plt.show()

'''
converted_images = convert_images(images)


image = get_biggest_digit(converted_images[16])
plt.imshow(image,cmap='gray')
plt.show()
image2 = get_biggest_digit2(converted_images[16])
plt.imshow(image2,cmap='gray')
plt.show()

