import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.applications import ResNet50
from keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from data_loader import load_dataset

labels_path = '../../dataset/qv_pipe_dataset/qv_pipe_train.json'
dataset_directory = '../dataset/qv_pipe_dataset/track1_raw_video/'
# frame_size = (480, 480)
# frames_per_video = 5

X_train, Y_train = load_dataset(labels_path,
                                dataset_directory,
                                frame_size=(480, 480),
                                frames_per_video=5)
input_shape = X_train.shape[1:]
output_shape = Y_train.shape[1]

# baseModel = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(480, 480, 3)))

# # Define the model
model = keras.Sequential([
    Input(shape=input_shape),
    Conv3D(64, kernel_size=(3, 3, 3), activation='relu'),
    MaxPooling3D(pool_size=(2, 2, 2)),
    Conv3D(128, kernel_size=(1, 3, 3), activation='relu'),
    MaxPooling3D(pool_size=(1, 2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(output_shape, activation='sigmoid')
])

# for layer in model.layers:
#     print(f"Layer: {layer.name}")
#     print(f"   Input Shape: {layer.input_shape}")
#     print(f"   Output Shape: {layer.output_shape}")

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Assuming you have preprocessed video clips (X_train) and corresponding labels (y_train) for training
# Set hyperparameters
batch_size = 32
epochs = 10

# Train the model
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

# Evaluate the model on the test data (assuming you have X_test and y_test)
test_loss, test_accuracy = model.evaluate(X_train, Y_train)

#
#
# # Assuming you have a dataset with video clips and their labels
# # X_train is a tensor of shape (num_samples, num_frames, height, width, channels)
# # y_train is a tensor of shape (num_samples, num_classes)
# model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size)
#
# # Evaluate the model
# test_loss, test_accuracy = model.evaluate(X_test, y_test)
