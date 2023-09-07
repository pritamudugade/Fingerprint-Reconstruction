# Fine Tune, create the new Autoencoder_weights.h5 file


import cv2
import os
import numpy as np
from glob import glob
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load and preprocess your dataset
# Set the path to the directory containing your JPEG format photos
data_dir = 'dataset/'

# Load and preprocess a single test image (replace 'test_image.jpg' with the path to your test image)
test_image_path = '10.jpg'
test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
test_image = cv2.resize(test_image, (224, 224))  # Resize to 224x224
test_image = test_image.astype('float32') / 255.0  # Rescale pixel values to [0, 1]
test_image = test_image.reshape(1, 224, 224, 1)  # Reshape for model compatibility

# Split the data into training and validation sets
# Replace this with your actual dataset loading and preprocessing
data = glob(os.path.join(data_dir, '*.jpg'))

images = []

def read_images(data):
    for i in range(len(data)):
        img = cv2.imread(data[i], cv2.IMREAD_GRAYSCALE)  # Read as grayscale
        img = cv2.resize(img, (224, 224))  # Resize to 224x224
        images.append(img)
    return images

images = read_images(data)

images_arr = np.asarray(images)
images_arr = images_arr.astype('float32')

# Rescale pixel values to the range [0, 1]
images_arr = images_arr / 255.0

# Reshape the data for compatibility with the model
images_arr = images_arr.reshape(-1, 224, 224, 1)

# Split the data into training and validation sets
train_X, valid_X = train_test_split(images_arr, test_size=0.2, random_state=13)

# The Convolutional Autoencoder
inChannel = 1

# Define the autoencoder model architecture
def autoencoder(input_img):
    # Encoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(encoded)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    return decoded

# Create the autoencoder model
input_img = Input(shape=(224, 224, inChannel))
autoencoder_model = Model(input_img, autoencoder(input_img))
autoencoder_model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error')

# Train the model from scratch
history = autoencoder_model.fit(train_X, train_X, batch_size=64, epochs=50)

# Save the weights after training
autoencoder_model.save_weights('autoencoder_weights.h5')

# Define a new instance of the autoencoder model for fine-tuning
fine_tune_model = Model(input_img, autoencoder(input_img))

# Load the pre-trained weights
fine_tune_model.load_weights('autoencoder_weights.h5')

# Compile the model for fine-tuning with a different learning rate
fine_tune_model.compile(optimizer=Adam(lr=0.0001), loss='mean_squared_error')

# Fine-tune the model on your dataset
fine_tune_history = fine_tune_model.fit(valid_X, valid_X, batch_size=64, epochs=20)

# Evaluate and visualize the fine-tuned model as needed
