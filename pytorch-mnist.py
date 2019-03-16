
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F 
import torch.utils.data
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

#Source: https://nextjournal.com/gkoehler/pytorch-mnist


# In[20]:


#Define hyper-parameters
n_epochs = 3
BATCH_SIZE = 32
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


# In[21]:


#Load the data
data_folder = './MNIST/data/'
train_filename = data_folder + 'train_images.pkl'
labels_filename = data_folder + 'train_labels.csv'
test_filename = data_folder + 'test_images.pkl'

train_images = pd.read_pickle(train_filename)
train_labels = pd.read_csv(labels_filename)
test_images = pd.read_pickle(test_filename)

train_labels = train_labels['Category'].values

X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.15)


# In[22]:


print(y_train.shape)
print(X_train.shape)


# In[23]:


#Transform data
#create feature and target tensor for train and test
torch_X_train = torch.from_numpy(X_train).type(torch.LongTensor)
torch_y_train = torch.from_numpy(y_train).type(torch.LongTensor) # data type is long
torch_X_test = torch.from_numpy(X_test).type(torch.LongTensor)
torch_y_test = torch.from_numpy(y_test).type(torch.LongTensor) # data type is long

torch_X_train = torch_X_train.view(-1, 1,64,64).float()
torch_X_test = torch_X_test.view(-1,1,64,64).float()
#train.size = torch.Size([34000, 1, 64, 64])
#test.size = torch.Size([6000, 1, 64, 64])

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(torch_X_train,torch_y_train)
test = torch.utils.data.TensorDataset(torch_X_test,torch_y_test)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = BATCH_SIZE, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = BATCH_SIZE, shuffle = False)


# In[34]:


#Build the model
#three 2D conv layer, 2 fully connected, ReLU, log softmax
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32,64, kernel_size=5)
        self.fc1 = nn.Linear(12*12*64, 256)
        #original : 3*3*64
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1,12*12*64 )
        #original -1, 3*3*64
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
 
cnn = CNN()
print(cnn)

it = iter(train_loader)
X_batch, y_batch = next(it)
print(cnn.forward(X_batch).shape)


# In[35]:


#Evaluation
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(X_train) for i in range(n_epochs + 1)]


def fit(model, train_loader):
    optimizer = torch.optim.Adam(model.parameters())#,lr=0.001, betas=(0.9,0.999))
    error = nn.CrossEntropyLoss()
    EPOCHS = n_epochs
    model.train()
    for epoch in range(EPOCHS):
        correct = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            var_X_batch = Variable(X_batch).float()
            var_y_batch = Variable(y_batch)
            optimizer.zero_grad()
            output = model(var_X_batch)
            loss = error(output, var_y_batch)
            loss.backward()
            optimizer.step()

            # Total correct predictions
            predicted = torch.max(output.data, 1)[1] 
            correct += (predicted == var_y_batch).sum()
            #print(correct)
            if batch_idx % 50 == 0:
                print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
                    epoch, batch_idx*len(X_batch), len(train_loader.dataset), 100.*batch_idx / len(train_loader), loss.data, float(correct*100) / float(BATCH_SIZE*(batch_idx+1))))
                train_losses.append(loss.item())
                train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))


# In[39]:


def evaluate(model):
    correct = 0 
    test_loss = 0
    for test_imgs, test_labels in test_loader:
        test_imgs = Variable(test_imgs).float()
        output = model(test_imgs)
        predicted = torch.max(output,1)[1]
        correct += (predicted == test_labels).sum()
        test_loss += F.nll_loss(output, test_labels, size_average=False).item()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
    print("Test accuracy:{:.3f}% ".format( float(correct) / (len(test_loader)*BATCH_SIZE)))


# In[37]:


fit(cnn,train_loader)
print("done")


# In[40]:


evaluate(cnn)


# In[41]:


fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
fig

