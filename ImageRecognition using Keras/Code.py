from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from matplotlib import pyplot as plt
from tensorflow import keras
import numpy as np
import cv2
import sys
from typing import Callable
from PyQt6 import QtGui, QtWidgets
from PyQt6.QtGui import QImage, QPainter, QPen, QMouseEvent
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtWidgets import QWidget, QGridLayout, QPushButton, QLineEdit
from PyQt6.QtGui import QPainter, QColorConstants

def load_dataset():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    num_pixels = X_train.shape[1] * X_train.shape[2]
    X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
    X_train = X_train / 255
    X_test = X_test / 255

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    return num_pixels, num_classes, X_train, X_test, y_train, y_test

def load_model(model):
    num_pixels, num_classes, X_train, X_test, y_train, y_test = load_dataset()
    print("\n##### Loading model from the file... #####\n")
    try:
        model = keras.models.load_model("3.Keras/model")

        return model, X_train, X_test, y_train, y_test

    except OSError:
        print("\n##### Model file not found, creating new model... #####\n")
        model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

        return model, X_train, X_test, y_train, y_test

def plot(history):
    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def image_classify(image, model):
    image_array = np.array([image])
    num_pixels = image_array.shape[1] * image_array.shape[2]
    image_array = image_array.reshape(image_array.shape[0], num_pixels).astype('float32')
    image_array = image_array / 255
    prediction = model.predict(image_array)

    return prediction

def qimage_to_array(image: QImage):
    image = image.convertToFormat(QImage.Format.Format_Grayscale8)
    ptr = image.bits()
    ptr.setsize(image.sizeInBytes())
    numpy_array = np.array(ptr).reshape(image.height(), image.width(), 1)

    cv2.imshow('Check if the function works!', numpy_array)
    return numpy_array

def predict(image: QImage, model):
    numpy_array = qimage_to_array(image)
    numpy_array = cv2.resize(numpy_array, (28,28))

    image_array = np.array([numpy_array])
    num_pixels = image_array.shape[1] * image_array.shape[2]
    image_array = image_array.reshape(image_array.shape[0], num_pixels).astype('float32')
    image_array = image_array / 255
    prediction = model.predict(image_array)

    return prediction.argmax()

PEN_WIDTH = 25
PEN_COLOR = QColorConstants.White
PIXMAP_SIZE = 256

class Paint(QtWidgets.QMainWindow):

    def __init__(self, predict_function: Callable = None):
        super().__init__()

        self.window = QWidget()

        self.paint = QtWidgets.QLabel()
        self.paint.setFixedWidth(PIXMAP_SIZE)
        self.paint.setFixedHeight(PIXMAP_SIZE)
        self.image = QImage(PIXMAP_SIZE, PIXMAP_SIZE, QImage.Format.Format_RGB32)
        self.image.fill(Qt.GlobalColor.black)

        self.layout = QGridLayout()
        self.prediction = QLineEdit()
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear)   

        self.layout.addWidget(self.prediction,1,0) 
        self.layout.addWidget(self.clear_button,1,1) 
        self.layout.addWidget(self.paint,0,0) 

        self.prediction.setDisabled(1)
        self.window.setLayout(self.layout)
        self.setCentralWidget(self.window)

        self.lastPoint = QPoint()

        self.predict_function = predict_function

    def clear(self):
        self.image.fill(QColorConstants.Black)
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            painter = QPainter(self.image)  
            painter.setPen(QPen(PEN_COLOR, PEN_WIDTH))
            painter.drawPoint(event.pos())
            self.drawing = True  
            self.lastPoint = event.pos()
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        if (event.buttons() & Qt.MouseButton.LeftButton) == Qt.MouseButton.LeftButton and self.drawing:
            painter = QPainter(self.image) 
            painter.setPen(QPen(PEN_COLOR, PEN_WIDTH))
            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button == Qt.MouseButton.LeftButton:
            self.drawing = False
        if self.predict_function:
            self.prediction.setText(str(self.predict_function(self.image)))
        
    def paintEvent(self, event: QtGui.QPaintEvent):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.paint.rect(), self.image, self.image.rect())

if __name__ == '__main__':
    model = Sequential()
    
    model, X_train, X_test, y_train, y_test = load_model(model)
       
    #history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=1)
    #scores = model.evaluate(X_test, y_test, verbose=0)

    #plot(history)
    #print("Baseline Error: %.2f%%" % (100-scores[1]*100))

    print("\n##### Saving model to the file... #####\n")
    model.save("3.Keras/model")

    image_paths = ["3.Keras/test_image1.jpg", "3.Keras/test_image2.jpg", "3.Keras/test_image3.jpg"]
    for i in range(len(image_paths)):
        print("\n##### Classifying image no. ", i + 1, " #####")
        image = plt.imread(image_paths[i])
        prediction = image_classify(image, model)
        print("Image classified as: ", prediction.argmax(), "\n")

    print("\n##### Openning painting window... #####\n")
    app = QtWidgets.QApplication(sys.argv)
    window = Paint(lambda x: predict(x, model))
    window.show()
    app.exec()