#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(42)


# In[2]:


# Loading the data (it's preloaded in Keras)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000)

print(x_train.shape)
print(x_test.shape)


# In[3]:


print(x_train[0])
print(y_train[0])


# In[4]:


#One-hot encoding the output into vector mode, each of length 1000
tokenizer = Tokenizer(num_words=1000)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
print(x_train[0])


# In[5]:


# One-hot encoding the output
num_classes = 2
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_train.shape)
print(y_test.shape)


# In[6]:


# TODO: Build the model architecture
model = Sequential()
model.add(Dense(512, activation ='relu', input_shape=(1000,)))
model.add(Dropout(0.5))
model.add(Dense(2, activation ='sigmoid'))


# TODO: Compile the model using a loss function and an optimizer.
model.compile(loss = 'binary_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])
model.summary()


# In[7]:


# TODO: Run the model. Feel free to experiment with different batch sizes and number of epochs.
model.fit(x_train, y_train, epochs = 15, batch_size = 32, verbose = 2, validation_data = (x_test, y_test))


# In[8]:


score = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: ", score[1])


# In[ ]:




