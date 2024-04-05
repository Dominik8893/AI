import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

from tensorflow.keras import datasets
(X_train,y_train),(X_test,y_test) = datasets.mnist.load_data()

X_train.shape, y_train.shape, X_test.shape, y_test.shape

X_train = X_train/255.
X_test = X_test/255.

plt.figure(figsize=(16,16))
for i,j in zip(np.random.randint(0,60000,15),np.arange(15)+1):
    plt.subplot(1,15,j)
    plt.imshow(X_train[i],cmap='gray_r')
    plt.axis('off')
plt.show()

net = Sequential()
net.add(Flatten(input_shape=(28,28)))          #add flatten layer
net.add(Dense(units=500, activation='relu'))   #add dense layer
net.add(Dropout(0.25))                         #add dropout layer
net.add(Dense(units=10, activation='softmax')) #add dense output layer

net.compile(optimizer='RMSprop',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

net.summary()

net.fit(X_train, y_train, epochs=5)
y_pred.shape, y_pred[0, :]
y_pred=y_pred.argmax(1)

y_pred
y_test

y_bad = y_test[y_pred!=y_test]
plt.hist(y_bad)

y_bad = np.arange(10000)
y_bad = y_bad[y_pred!=y_test]

plt.figure(figsize=(8,8))
for i,j in zip(y_bad[:25],np.arange(25)+1):
    plt.subplot(5,5,j)
    plt.imshow(X_test[i],cmap='gray_r')
    plt.axis('off')
    plt.title('%1d/%1d' %(y_pred[i],y_test[i]),fontsize=10)
plt.show()