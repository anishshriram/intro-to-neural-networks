import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''
After a certain number of epochs, we see that the loss starts to vary / go up and down. To mitigate this, as well as
determine when to stop the epochs, this new class can be added. Note the changes on the model.fit()
'''

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') < 0.4:
            print("\nLoss is low so cancelling training!")
            self.model.stop_training = True

# Instantiate the above class
callbacks = MyCallback()

'''
Coding a scenario in which the neural network can recognize different items of clothing.

Uses dataset called Fashion MNIST
    Collection of 70,000 images of 10 types of clothing
    28x28 array of greyscales
    type of clothing labeled by a number (to get rid of lang biases)

First 60,000 images to train neural network, last 10,000 to test it
'''

# Loading the Fashion MNIST dataset
mnist = tf.keras.datasets.fashion_mnist

import requests

requests.packages.urllib3.disable_warnings()
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

# Giving two sets of lists, for training and testing values for the graphics that contain the clothing items and labels
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

'''
# By running the following, you can see what the above values look like for an image, as well as the actual image
np.set_printoptions(linewidth=200)
plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])
'''

# Normalizing (instead of the numbers being from 0 -> 255, make it from 0-1)
# In python, you don't have to loop through the list - just divide entire thing
training_images = training_images / 255.0
test_images = test_images / 255.0

'''
For the model:
Sequential: Defines the SEQUENCE of layers in the neural network
Flatten: Images are a square, flatten takes the square and turns it into a one dimensional set
Dense: Adds a layer of neurons
    Needs activation functions to tell them what to do
    Relu: Basically, if X > 0 return X, else return 0
    Softmax: Takes a set of values, and picks out the biggest one
'''
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# Now actually build the model. Compiling with the optimizer and loss function as before
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the neural network by calling model.fit()
# Low amount of epochs because it will take a long time otherwise
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])

# Actually seeing the accuracy value after the final epoch
model.evaluate(test_images, test_labels)
