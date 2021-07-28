import os
import zipfile

# allows me to unzip the data

local_zip = '/Users/anishshriram/Downloads/horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/Users/anishshriram/Downloads/horse-or-human')
zip_ref.close()

# Directory with the training images (horses)
train_horse_dir = os.path.join('/Users/anishshriram/Downloads/horse-or-human/horses')

# Directory wiht the training images (humans)
train_human_dir = os.path.join('/Users/anishshriram/Downloads/horse-or-human/humans')

print('total training horse images:', len(os.listdir(train_horse_dir)))
print('total training human images:', len(os.listdir(train_human_dir)))

# Building the Neural Network to train and test

import tensorflow as tf

'''
We will use 3 convolutional layers
Because two class classification problem (in other words it is binary) end the network with a sigmoid activation
that way the output wil be either 0 or 1
'''

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other (
    # 'humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
])

'''
Training model with the binary_crossentropy loss function for the same reason we use sigmoid activation
We used Adam optimizer before, which would work just as well, RMSprop algorithm, is just preferred to this scenario
because it is a stochastic gradient descent (SGD)
'''

from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])

# Data Prepossessing

# Rescaling the date and normalizing it so that it is easier processed in the neural network

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        '/Users/anishshriram/Downloads/horse-or-human',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 150x150
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

# Training the model - trying for about 15 epochs

history = model.fit(
      train_generator,
      steps_per_epoch=8,
      epochs=15,
      verbose=1)