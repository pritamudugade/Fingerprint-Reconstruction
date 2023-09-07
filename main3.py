# 5 outputs at end


import cv2
import os
import numpy as np
from glob import glob
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Set the path to the directory containing your JPEG format photos
data_dir = 'dataset/'

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

# Load your test data and preprocess it (similar to what you did with training and validation data)
test_data_dir = 'dataset/'  # Replace with the path to your test dataset directory

test_data = glob(os.path.join(test_data_dir, '*.jpg'))

test_images = []

def read_test_images(test_data):
    for i in range(len(test_data)):
        img = cv2.imread(test_data[i], cv2.IMREAD_GRAYSCALE)  # Read as grayscale
        img = cv2.resize(img, (224, 224))  # Resize to 224x224
        test_images.append(img)
    return test_images

test_images = read_test_images(test_data)

test_X = np.asarray(test_images)
test_X = test_X.astype('float32')

# Rescale pixel values to the range [0, 1]
test_X = test_X / 255.0

# Reshape the test data for compatibility with the model
test_X = test_X.reshape(-1, 224, 224, 1)

# The Convolutional Autoencoder
batch_size = 256
epochs = 500
inChannel = 1
x, y = 224, 224
input_img = Input(shape=(x, y, inChannel))

# Define the autoencoder model
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
input_img = Input(shape=(224, 224, 1))
autoencoder_model = Model(input_img, autoencoder(input_img))
autoencoder_model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error')

# Define callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(factor=0.2, patience=5)

# Train the autoencoder
history = autoencoder_model.fit(train_X, train_X,
                                batch_size=64,
                                epochs=50,
                                validation_data=(valid_X, valid_X),
                                callbacks=[early_stopping, reduce_lr])

# Evaluate the model
loss = autoencoder_model.evaluate(valid_X, valid_X)
print("Validation Loss:", loss)

# Reconstruct test images
reconstructed_images = autoencoder_model.predict(test_X)

# Display some original and reconstructed images
n = 5  # Number of images to display
plt.figure(figsize=(10, 4))

for i in range(n):
    # Original Image
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(test_X[i].reshape(224, 224), cmap='gray')
    plt.title("Original")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # Reconstructed Image
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstructed_images[i].reshape(224, 224), cmap='gray')
    plt.title("Reconstructed")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
