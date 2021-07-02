import tensorflow as tf
import numpy as np
from tensorflow import keras

'''
This is the 'hello world' program of neural networks
Feeding the network a set of X values and Y values, it has to find the relationship between them
'''

# This network has 1 layer, and that layer only has 1 neuron, and that neuron's shape is 1, because it takes in 1 value
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# The loss function compares the guessed answer to the known correct answers and compares how good or bad it did
# The optimizer function makes another guess, based on how good or bad the loss function says it did
model.compile(optimizer='sgd', loss='mean_squared_error')

# Data to be inputted
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Epochs are the number of times it will run, in this case, 500 times
model.fit(xs, ys, epochs=500)

# Printing a guess as to what an X value of 10 will output. It should be 19, but the network prints out something very
# close to that number, but not exact. This is because the neural network deals only in probabilities - it guessed that
# the relationship was most likely y = 2x-1, but it doesn't knw for sure. As such, it prints out the most realistic
# or probable number
print(model.predict([10.0]))
