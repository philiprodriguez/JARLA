import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from matplotlib import pyplot
from matplotlib import image

print("JARLA is starting...")

# Build the JARLA neural network model...

input_layer = layers.Input(shape=(512, 512, 3))
x = layers.Conv2D(32, kernel_size=(5, 5), padding='same', activation='relu')(input_layer)
x = layers.Conv2D(32, kernel_size=(5, 5), padding='same', activation='relu')(x)
x = layers.Conv2D(32, kernel_size=(5, 5), padding='same', activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2, 2), padding='valid')(x)
x = layers.Conv2D(16, kernel_size=(5, 5), padding='same', activation='relu')(x)
x = layers.Conv2D(16, kernel_size=(5, 5), padding='same', activation='relu')(x)
x = layers.Conv2D(16, kernel_size=(5, 5), padding='same', activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2, 2), padding='valid')(x)
x = layers.Conv2D(8, kernel_size=(5, 5), padding='same', activation='relu')(x)
x = layers.Conv2D(8, kernel_size=(5, 5), padding='same', activation='relu')(x)
x = layers.Conv2D(8, kernel_size=(5, 5), padding='same', activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2, 2), padding='valid')(x)
x = layers.Conv2D(4, kernel_size=(5, 5), padding='same', activation='relu')(x)
x = layers.Conv2D(4, kernel_size=(5, 5), padding='same', activation='relu')(x)
x = layers.Conv2D(4, kernel_size=(5, 5), padding='same', activation='relu')(x)

x = layers.Flatten()(x)

x = layers.Dense(1024, activation='relu')(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dense(4, activation='linear')(x)

model = Model(inputs=input_layer, outputs=x)

print(model.summary())

image = image.imread("image.jpg")
print(image.shape)
pyplot.imshow(image)
pyplot.show()