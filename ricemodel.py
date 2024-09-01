import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from keras import regularizers,optimizers
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adamax 
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import cv2

# Load the dataset
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    r"C:\Users\hp\Desktop\New folder\Rice Leaf Disease Images",
    shuffle=True,
    image_size=(150, 150),
    batch_size=32
)

# Define the path
base_dir = r'C:\Users\hp\Desktop\New folder\Rice Leaf Disease Images'

# Initialize the data generator
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.1,  # reserve 10% of the data for validation
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Generate training data
train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    color_mode="rgb",
    batch_size=32,
    subset='training',  # set as training data
    class_mode='categorical'
)

# Generate validation data
validation_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    color_mode="rgb",
    batch_size=32,
    subset='validation',  # set as validation data
    class_mode='categorical'
)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPool2D((2, 2)),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPool2D((2, 2)),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPool2D((2, 2)),
    BatchNormalization(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),
    Dense(len(train_generator.class_indices), activation='softmax')  # Number of classes
])

# Compile the model
model.compile(
    optimizer=Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=validation_generator
)

# Plotting the results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(25)

plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

import json

# Save the model
model.save('rice_leaf_disease_model.h5')

# Save the class indices
class_indices = train_generator.class_indices
with open('class_indices.json', 'w') as f:
    json.dump(class_indices, f)

