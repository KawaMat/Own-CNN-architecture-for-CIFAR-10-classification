import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), (X_test,y_test) = datasets.cifar10.load_data() #loading data for training

X_train = X_train / 255.0 #normalization of size from 1-255 to 0-1
X_test = X_test / 255.0

MyModel = models.Sequential([ #model construction
    layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Dropout(0.2),

    layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
    layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Dropout(0.2),

    layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Dropout(0.2),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

MyModel.compile(optimizer='adamax',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

MyModel.fit(X_train, y_train, epochs=12, batch_size=32)


y_pred = MyModel.predict(X_test)
print("Acc value for testing set -",int((MyModel.evaluate(X_test,y_test)[1]*100)//1),"%")