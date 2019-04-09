# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 14:48:15 2019

@author: ylb18188
"""
#start off by importing the classes and functions required for this model and initializing the random number generator to a constant value to ensure we can easily reproduce the results.

# MLP for the IMDB problem
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
#Next we will load the IMDB dataset. We will simplify the dataset as discussed during the section on word embeddings. Only the top 5,000 words will be loaded.
#We will also use a 50%/50% split of the dataset into training and test. This is a good standard split methodology
# load the dataset but only keep the top n words, zero the rest #We will bound reviews at 500 words, truncating longer reviews and zero-padding shorter reviews
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)
#Now we can create our model. We will use an Embedding layer as the input layer, setting the vocabulary to 5,000, the word vector size to 32 dimensions and the input_length to 500. The output of this first layer will be a 32×500 sized matrix as discussed in the previous section.
#We will flatten the Embedded layers output to one dimension, then use one dense hidden layer of 250 units with a rectifier activation function. The output layer has one neuron and will use a sigmoid activation to output values of 0 and 1 as predictions.
#The model uses logarithmic loss and is optimized using the efficient ADAM optimization procedure.
#We can fit the model and use the test set as validation while training. This model overfits very quickly so we will use very few training epochs, in this case just 2.
#There is a lot of data so we will use a batch size of 128. After the model is trained, we evaluate its accuracy on the test dataset.
# create the model
model = Sequential()
model.add(Embedding(top_words, 32, input_length=max_words))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
#We can fit the model and use the test set as validation while training. This model overfits very quickly so we will use very few training epochs, in this case just 2.
#There is a lot of data so we will use a batch size of 128. After the model is trained, we evaluate its accuracy on the test dataset.
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
#Test loss: 0.30056160572052004 #Test accuracy: 0.87312
scores1 = model.evaluate(X_train, y_train, verbose=0)
print('Training loss:', scores1[0])
print('Training accuracy:', scores1[1])
#Training loss: 0.076193997797966 #Training accuracy: 0.98508































