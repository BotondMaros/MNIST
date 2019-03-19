from __future__ import print_function
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle as pkl
import time
import copy
import math
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F 
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from resnet152 import ResNet152, Bottleneck

data_folder = './data/'
kaggle_filename = data_folder + 'test_images.pkl'
kaggle_images = pd.read_pickle(kaggle_filename)

'''
f = lambda x: 0 if x < 255 else 255
vf = np.vectorize(f)
for i in range(len(kaggle_images)):
    for j in range(len(kaggle_images[i])):
        vf(kaggle_images[i][j])
'''

torch_kaggle = torch.from_numpy(kaggle_images).type(torch.LongTensor)
torch_kaggle = torch_kaggle.view(-1, 1,64,64).float()

kaggle = torch.utils.data.TensorDataset(torch_kaggle)
kaggle_loader = torch.utils.data.DataLoader(kaggle, batch_size = 1, shuffle = False)

model = ResNet152(Bottleneck, [3, 8, 36, 3])

model.load_state_dict(torch.load('./resnet_model_152.ckpt'))
print(model)

predictions = []
model.eval()
for x in kaggle_loader:
    outputs = model(x[0])
    predicted = torch.max(outputs.data, 1)
    #print(predicted[1].item())
    predictions.append(predicted[1].item())
    #print(predictions)
    if (len(predictions) % 1000) == 0:
        print(len(predictions))

x = [x for x in range(0,10000)]
df = pd.DataFrame(data={"ID": x, "Category": predictions})
df.to_csv("./submission_final.csv", sep=',',index=False)