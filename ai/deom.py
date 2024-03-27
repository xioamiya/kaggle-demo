#量标准化
import numpy as np
from matplotlib import pyplot as plt

from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPool2D
from keras.layers import BatchNormalization

from keras.datasets import cifar10
(X_train, y_train), (X_val, y_val) = cifar10.load_data()

X_train = X_train.astype('float32')/255.
X_val = X_val.astype('float32')/255.
y_train.shape

n_classes = 10
y_train = np_utils.to_categorical(y_train, n_classes)
y_val  = np_utils.to_categorical(y_val, n_classes)


input_shape = X_train[0].shape
input_shape

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), input_shape= input_shape, padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(32, kernel_size=(3,3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2),padding='same'))

model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3,3)))
model.add(Activation('relu'))
model.add(Conv2D(64, kernel_size=(3,3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2),padding='same'))

model.add(Dropout(0.25))


model.add(Conv2D(128, kernel_size=(3,3)))
model.add(Activation('relu'))
model.add(Conv2D(128, kernel_size=(3,3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2),padding='same'))

model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(n_classes, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])


callbacks = [EarlyStopping(monitor='val_accuracy', patience=5,verbose=1)]


batch_sise = 128
n_epochs = 200
history = model.fit(X_train, y_train, batch_size=batch_sise, epochs=n_epochs,verbose=1, validation_data=(X_val, y_val),callbacks=callbacks)