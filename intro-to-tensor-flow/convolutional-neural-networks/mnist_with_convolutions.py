import tensorflow as tf
from os import path, getcwd, chdir

# code is essentially the same as handwriting_recog --> but greater accuracy was achieved using convolutions and pooling
'''
Convolutions - Passing filters over the image. These filters are arrays, and the pixels are reformatted according to 
that array. 

https://en.wikipedia.org/wiki/Kernel_(image_processing)

For example, if you look at the above link, you'll see a 3x3 that is defined for edge detection where the middle cell 
is 8, and all of its neighbors are -1. In this case, for each pixel, you would multiply its value by 8, then subtract 
the value of each neighbor. Do this for every pixel, and you'll end up with a new image that has the edges enhanced.

Pooling - compressing the image while maintaining the features that were hilighted by the convolutions. For this
assignment I will use 2x2 pooling, which takes a 2x2 array of pixels, and picks the biggest one. This way it turns 4
pixels into 1. It reduces the image by 25%

'''

path = '/Users/anishshriram/Downloads/mnist.npz'

DESIRED_ACCURACY = 0.998


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epochs, logs={}):
        if (logs.get('acc') is not None and logs.get('acc') >= DESIRED_ACCURACY):
            print('\nReached 99.8% accuracy so cancelling training!')
            self.model.stop_training = True


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


def train_mnist_conv():
    # Please write your code only where you are indicated.
    # please do not remove model fitting inline comments.

    # YOUR CODE STARTS HERE

    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data(path=path)
    training_images = training_images.reshape(60000, 28, 28, 1)
    training_images = training_images / 255.0
    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images = test_images / 255.0

    '''
    Instead of an input layer, I added a convolution. The paramaters are:
    1. The number of convolutions to create. People usually start with something in the order of 32
    2. THe size of the convolution - I did 3x3 grid
    3. The activation function, used relu just like before
    4. And only in the first layer, the shape of the input data
    
    The MaxPooling works just as described above
    '''

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    callbacks = myCallback()
    # model fitting
    history = model.fit(training_images, training_labels, epochs=20, callbacks=[callbacks])
    # model fitting
    return history.epoch, history.history['acc'][-1]

    # YOUR CODE ENDS HERE


_, _ = train_mnist_conv()
