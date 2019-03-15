
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
    return((images*255).astype(np.uint8))

def get_biggest_digit(image):
    blur = cv2.GaussianBlur(image,(7,7),0)
    _,img_bin = cv2.threshold(blur,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours,_ = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    countours_largest = sorted(contours, key=lambda x: cv2.contourArea(x))[-2]
    # filter everything outside contour?
    bb=cv2.boundingRect(countours_largest)
    # return numpy array of single biggest digit on uniform background
    return


images = get_pickle_data(train_filename)
labels = get_labels_from_csv()

image = (images[16]*255).astype(np.uint8)
plt.imshow(image,cmap='gray')
plt.show()

blur = cv2.GaussianBlur(image,(7,7),0)
ret3,img_bin = cv2.threshold(blur,128,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.imshow(img_bin,cmap='gray')
plt.title('Threshold: ')
plt.show()

contours,_ = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
countours_largest = sorted(contours, key=lambda x: cv2.contourArea(x))[-2]
bb=cv2.boundingRect(countours_largest)


white = (np.zeros(image.shape)).astype(np.uint8)
mask = cv2.drawContours(white, contours,-2,255,-1)
out = cv2.bitwise_and(img_bin, mask)
plt.imshow(white,cmap='gray')
plt.title('Filter: ')
plt.show()

'''
pt1=(bb[0],bb[1]) # upper coordinates 
pt2=(bb[0]+bb[2],bb[1]+bb[3]) # lower coordinates
img_gray_bb=image.copy()
cv2.rectangle(img_gray_bb,pt1,pt2,255,1)
plt.imshow(img_gray_bb,cmap='gray')
plt.show()
'''
