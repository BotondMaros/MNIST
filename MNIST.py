
# coding: utf-8

# ### Load data

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')


# In[2]:


train_images = pd.read_pickle('./data/train_images.pkl')
train_labels = pd.read_csv('./data/train_labels.csv')


# In[3]:


train_images.shape


# In[4]:


import matplotlib.pyplot as plt

#Let's show image with id 16
img_idx = 16

plt.title('Label: {}'.format(train_labels.iloc[img_idx]['Category']))
plt.imshow(train_images[img_idx])
plt.show()


# In[5]:


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


# ### Splitting and reshaping the data.
# #### In the case of RGB, the first dimension pixels would be 3 for the red, green and blue components and it would be like having 3 image inputs for every color image. 

# In[8]:


#Splitting the data
X_train,  X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.2)
# reshape to be [samples][pixels][width][height]
#X_train = img_to_array(X_train)
#X_test = img_to_array(X_test)
X_train = X_train.reshape(X_train.shape[0], 1, 64, 64)
X_test = X_test.reshape(X_test.shape[0], 1, 64, 64)
#scaling
X_train = X_train/255
X_test = X_test/255
#encoding the labels
y_train = np_utils.to_categorical(y_train['Category'])
y_test = np_utils.to_categorical(y_test['Category'])
num_classes = y_test.shape[1]


# ### Creating and bulding the model

# In[9]:


def baseline_model():
	# create model
	model = Sequential()
	model.add(Conv2D(32, (5, 5), input_shape=(1, 64, 64), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


# In[ ]:


# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))

