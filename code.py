import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
BATCH_SIZE = 32
EPOCHS = 10

# Define data directories
train_dir = '/content/drive/MyDrive/Forest Fire Dataset/Training'
test_dir = '/content/drive/MyDrive/Forest Fire Dataset/Testing'

# Define data generators with data augmentation
# process of artificially generating new data from existing data, primarily to train new machine learning (ML) models.

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,

    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
# The testing data generator only rescales the pixel values.
test_datagen = ImageDataGenerator(rescale=1./255)


# Generate batches of training and testing data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=test_generator,
    validation_steps=test_generator.n // BATCH_SIZE
)

# Save the trained model
model.save('forest_fire_detection_model.h5')

