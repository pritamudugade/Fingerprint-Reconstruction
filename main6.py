# Evaluate model

# Fine Tune, create the new Autoencoder_weights.h5 file

import cv2
import os
import numpy as np
from glob import glob
import skimage
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

# Define the input layer
input_img = Input(shape=(224, 224, inChannel))

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

# Reconstruct some validation images using the fine-tuned model
num_samples_to_visualize = 4  # Number of samples to visualize
reconstructed_images = fine_tune_model.predict(valid_X[:num_samples_to_visualize])

# Display original and reconstructed images
plt.figure(figsize=(12, 4))
for i in range(num_samples_to_visualize):
    # Original Image
    ax = plt.subplot(2, num_samples_to_visualize, i + 1)
    plt.imshow(valid_X[i].reshape(224, 224), cmap='gray')
    plt.title("Original")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Reconstructed Image
    ax = plt.subplot(2, num_samples_to_visualize, i + 1 + num_samples_to_visualize)
    plt.imshow(reconstructed_images[i].reshape(224, 224), cmap='gray')
    plt.title("Reconstructed")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Calculate PSNR and SSIM for a reconstructed image
psnr = peak_signal_noise_ratio(valid_X[0].reshape(224, 224), reconstructed_images[0].reshape(224, 224))
ssim = structural_similarity(valid_X[0].reshape(224, 224), reconstructed_images[0].reshape(224, 224))

print("PSNR:", psnr)
print("SSIM:", ssim)
