from __future__ import absolute_import, division, print_function

"""
Source: https://gist.github.com/AFAgarap/326af55e36be0529c507f1599f88c06e
adapted by: matias@u.nus.edu
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
print(tf.__version__)

# TODO: make it all one object
# use 3 or 2 layers
# leaky relu

class Encoder(tf.keras.layers.Layer):
    def __init__(self, intermediate_dim):
        super(Encoder, self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(units=intermediate_dim, activation=tf.nn.relu)
        self.output_layer = tf.keras.layers.Dense(units=intermediate_dim, activation=tf.nn.relu)
    
    def call(self, input_features):
        activation = self.hidden_layer(input_features)
        return self.output_layer(activation)
    
class Decoder(tf.keras.layers.Layer):
    def __init__(self, intermediate_dim, original_dim):
        super(Decoder, self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(units=intermediate_dim, activation=tf.nn.relu)
        self.output_layer = tf.keras.layers.Dense(units=original_dim, activation=tf.nn.relu)
    
    def call(self, code):
        activation = self.hidden_layer(code)
        return self.output_layer(activation)
    
class Autoencoder(tf.keras.Model):
    def __init__(self, intermediate_dim, original_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(intermediate_dim=intermediate_dim)
        self.decoder = Decoder(intermediate_dim=intermediate_dim, original_dim=original_dim)
        
        
    def call(self, input_features):
        code = self.encoder(input_features)
        reconstructed = self.decoder(code)
        return reconstructed

def loss(model, original):
    return tf.reduce_mean(tf.square(tf.subtract(model(original), original))) # L2

# This annotation causes the function to be "compiled" and therefore run as graph
def train(loss, model, opt, original):
    with tf.GradientTape() as tape:
        gradients = tape.gradient(loss(model, original), model.trainable_variables)
    gradient_variables = zip(gradients, model.trainable_variables)
    opt.apply_gradients(gradient_variables)



np.random.seed(1)
tf.random.set_seed(1)
batch_size = 128
epochs = 10
learning_rate = 1e-3
momentum = 9e-1
intermediate_dim = 64
original_dim = 784

autoencoder = Autoencoder(intermediate_dim=64, original_dim=784)
opt = tf.optimizers.Adam(learning_rate=learning_rate)

(training_features, _), (test_features, _) = tf.keras.datasets.mnist.load_data()
training_features = training_features / np.max(training_features)
training_features = training_features.reshape(training_features.shape[0],
                                              training_features.shape[1] * training_features.shape[2]).astype(np.float32)
training_dataset = tf.data.Dataset.from_tensor_slices(training_features).batch(batch_size)

writer = tf.summary.create_file_writer('/Users/matias/Downloads/tmp')

with writer.as_default():
    with tf.summary.record_if(True):
        for epoch in range(epochs):
            for step, batch_features in enumerate(training_dataset):
                train(loss, autoencoder, opt, batch_features)
                loss_values = loss(autoencoder, batch_features)
                original = tf.reshape(batch_features, (batch_features.shape[0], 28, 28, 1))
                reconstructed = tf.reshape(autoencoder(tf.constant(batch_features)), (batch_features.shape[0], 28, 28, 1))
                tf.summary.scalar('loss', loss_values, step=step)
                tf.summary.image('original', original, max_outputs=10, step=step)
                tf.summary.image('reconstructed', reconstructed, max_outputs=10, step=step)