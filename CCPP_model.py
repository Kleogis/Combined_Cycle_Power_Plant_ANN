# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 19:07:58 2021

@author: Kleogis
"""

#importing libraries
import numpy as np
import pandas as pd
import tensorflow as tf



#Data processing
#importing the dataset

dataset = pd.read_excel('Folds5x2_pp.xls')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Splitting the dataset into the training and test sets

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#initializing the ANN

ann = tf.keras.models.Sequential()

#Adding the input layer and the first hidden layer

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

#Adding the second hidden layer

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

#Adding the output layer

ann.add(tf.keras.layers.Dense(units=1))

#Compiling the ANN

ann.compile(optimizer = 'adam', loss = 'mean_squared_error')

#training the ANN model on the Training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 10)

#Predicting the results of the Test set
y_pred = ann.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))





