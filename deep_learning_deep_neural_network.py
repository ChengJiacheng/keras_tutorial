import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Flatten, Dense, Conv2D
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical

from keras.datasets import cifar10

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
tf.test.is_built_with_cuda()

# tf.device('/gpu:0')
tf.device('/cpu:0')

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


NUM_CLASSES = 10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

x_train[54, 12, 13, 1]


input_layer = Input((32,32,3))

x = Flatten()(input_layer)

x = Dense(512, activation = 'relu')(x)
x = Dense(128, activation = 'relu')(x)

output_layer = Dense(NUM_CLASSES, activation = 'softmax')(x)

model = Model(input_layer, output_layer)

model.summary()

opt = Adam(lr=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])



model.fit(x_train
          , y_train
          , batch_size=64
          , epochs=15
          , shuffle=True)

model.evaluate(x_test, y_test)
