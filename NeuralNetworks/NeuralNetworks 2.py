import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

from tensorflow.keras import datasets

(X_train,y_train), (X_test,y_test) = datasets.cifar10.load_data()

X_train = X_train/255.
X_test = X_test/255.

class_names = ["plane", "car", "bird", "cat", "deer", "doggo", "frog", "horse", "boat", "truck"]

plt.figure(figsize=(8,8))
for i,j in zip(np.random.randint(0,50000,25),np.arange(25)+1):
    plt.subplot(5,5,j)
    plt.imshow(X_train[i],cmap='gray_r')
    plt.axis('off')
    plt.title(class_names[y_train[i][0]],fontsize=12)
plt.show()

net = Sequential()
net.add(Conv2D(32, 3, activation = "relu", input_shape = X_train.shape[1:]))
net.add(MaxPooling2D())
net.add(Conv2D(64, 3, activation = "relu"))
net.add(MaxPooling2D())
net.add(Conv2D(64, 3, activation = "relu"))

net.summary()

net.add(Flatten())
net.add(Dense(64, activation = "relu"))
net.add(Dense(10))

net.summary()

net.compile(optimizer='adam',
            loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

training_result = net.fit(X_train, y_train, epochs = 7, validation_data = (X_test, y_test))
y_pred = np.argmax(net.predict(X_test),axis=-1)

y_test = y_test.reshape(10000,)

ind = y_pred != y_test
y_bad = np.arange(10000)
y_bad = y_bad[ind]
plt.figure(figsize=(8,8))
for i,j in zip(y_bad[:25],np.arange(25)+1):
    plt.subplot(5,5,j)
    plt.imshow(X_test[i],cmap='gray_r')
    plt.axis('off')
    plt.title('%s/%s' %(class_names[y_pred[i]],class_names[y_test[i]]),fontsize=12)
plt.show()

plt.plot(training_result.history['accuracy'],label='accuracy')
plt.plot(training_result.history['val_accuracy'],label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.4,1])
plt.legend(loc='lower right')

yy_pred = net.predict(X_test)
plt.figure(figsize=(8,3))
ind = np.random.randint(0,len(y_bad),1)[0]
plt.barh(np.arange(10),yy_pred[y_bad[ind],:])
plt.yticks(ticks=[0,1,2,3,4,5,6,7,8,9],labels=class_names)
plt.title('real class is '+str(class_names[y_test[y_bad[ind]]]))