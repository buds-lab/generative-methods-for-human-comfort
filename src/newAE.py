from __future__ import absolute_import, division, print_function, unicode_literals

"""
Source: https://www.tensorflow.org/beta/tutorials/generative/cvae
adapted by: matias@u.nus.edu
"""

# Tensorflow
import tensorflow as tf
from tensorflow.keras import layers
print(tf.__version__)

# Progress bar
from tqdm import tqdm

# Numpy, pandas, matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import PIL
import imageio
from IPython import display
import os
import glob


from IPython import display
# Sklearn
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# TODO: 
# - use same structure as GAN notebook
# - try relu AND leaky relu
# - try with BatchNormalization

class AE(tf.keras.Model):
    def __init__(self, data_dim=15, n_hidden=512, n_layers=4, display=False):
        super(AE, self).__init__()
        self.display = display
        self.scaler = None
        self.columns = []
        
        self.data_dim = data_dim # number of features
        self.latent_dim = 5 # TODO: CHANGE, try different
        self.n_hidden = n_hidden
        self.n_layers = n_layers
                    
        # Build the encoder and decoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
    
    def build_encoder(self):
        """
        """
        n_hidden= self.n_hidden
        n_layers = self.n_layers

        model = tf.keras.Sequential(name='encoder')

        if n_layers == 1:
            model.add(layers.Dense(n_hidden, input_dim=self.data_dim, activation=tf.nn.relu))
#         model.add(layers.BatchNormalization())
            #model.add(layers.LeakyReLU(alpha=0.2))
            self.latent_dim = n_hidden
        
        else:
            # hidden layers
            for layer in range(n_layers):
                if layer == 0: # first layer
                    model.add(layers.Dense(n_hidden, use_bias=False, input_dim=self.data_dim))
        #             model.add(layers.BatchNormalization())
                    model.add(layers.LeakyReLU(alpha=0.2))
                    n_hidden /= 2 # decrease size by half
    
                # remaining layers
                self.latent_dim = n_hidden
                model.add(layers.Dense(n_hidden))
        #         model.add(layers.BatchNormalization())
                model.add(layers.LeakyReLU(alpha=0.2))
                n_hidden /= 2 # decrease size by half
    
    
        model.summary()
        return model
    
    def build_decoder(self):
        """
        """
        n_hidden= self.latent_dim * 2
        n_layers = self.n_layers 

        model = tf.keras.Sequential(name='decoder')

        if n_layers == 1:
            model.add(layers.Dense(self.data_dim, input_dim=self.latent_dim, activation=tf.nn.relu))
#         model.add(layers.BatchNormalization())
            #model.add(layers.LeakyReLU(alpha=0.2))
        else: 
            # hidden layers
            for layer in range(n_layers - 1):
                if layer == 0: # first layer
                    model.add(layers.Dense(n_hidden, input_dim=self.latent_dim)) # TODO: activation=tf.nn.relu
        #             model.add(layers.BatchNormalization())
                    model.add(layers.LeakyReLU(alpha=0.2))
                    n_hidden *= 2 # double size
    
                # remaining layers
                model.add(layers.Dense(n_hidden))
        #         model.add(layers.BatchNormalization())
                model.add(layers.LeakyReLU(alpha=0.2))
                n_hidden *= 2 # double size
    
            # last layer
            model.add(layers.Dense(self.data_dim))
    #         model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU(alpha=0.2))
        
        model.summary()
        return model

    def encode(self, x):
        encoded = self.encoder(x)
        return encoded

    def decode(self, z):
        decoded = self.decoder(z)
        return decoded

@tf.function
def compute_loss(model, x): # TODO: use L2 loss
    encoded = model.encode(x)
    decoded = model.decode(encoded)
    return tf.reduce_mean(tf.square(tf.subtract(decoded, x))) # L2

@tf.function
def compute_apply_gradients(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
def generate_data(model, epoch, test_input):
    predictions = model.sample(test_input)
    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

def train(model, train_dataset, test_dataset, optimizer, EPOCHS):
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        for train_x in train_dataset:
            compute_apply_gradients(model, train_x, optimizer)
        end_time = time.time()

        if epoch % 1 == 0:
            loss = tf.keras.metrics.Mean()
            for test_x in test_dataset:
                loss(compute_loss(model, test_x))
            elbo = -loss.result()
            display.clear_output(wait=False)
            print('Epoch: {}, Test set ELBO: {}, '
            'time elapse for current epoch {}'.format(epoch, elbo, end_time - start_time))

def train_tensorboard(model, train_dataset, optimizer, EPOCHS):
    writer = tf.summary.create_file_writer('/Users/matias/Downloads/tmp2')

    with writer.as_default():
        with tf.summary.record_if(True):
            for epoch in range(EPOCHS):
                for step, train_x in enumerate(train_dataset):
                    compute_apply_gradients(model, train_x, optimizer)
                    loss_values = compute_loss(model, train_x)
                    original = tf.reshape(train_x, (train_x.shape[0], 28, 28, 1))
                    
                    # autoencoder
                    encoded = model.encode(train_x)
                    decoded = model.decode(encoded)
                    
                    reconstructed = tf.reshape(decoded, (decoded.shape[0], 28, 28, 1))
                    
                    tf.summary.scalar('loss', loss_values, step=step)
                    tf.summary.image('original', original, max_outputs=10, step=step)
                    tf.summary.image('reconstructed', reconstructed, max_outputs=10, step=step)
                    
# parameters
np.random.seed(1)
tf.random.set_seed(1)
batch_size = 128
epochs = 50
learning_rate = 1e-3
intermediate_dim = 64
original_dim = 784

(training_features, _), (test_features, _) = tf.keras.datasets.mnist.load_data()
training_features = training_features / np.max(training_features)
training_features = training_features.reshape(training_features.shape[0],
                                              training_features.shape[1] * training_features.shape[2]).astype(np.float32)
training_dataset = tf.data.Dataset.from_tensor_slices(training_features).batch(batch_size)

test_features = test_features / np.max(test_features)
test_features = test_features.reshape(test_features.shape[0],
                                              test_features.shape[1] * test_features.shape[2]).astype(np.float32)
test_features = tf.data.Dataset.from_tensor_slices(training_features).batch(batch_size)

optimizer = tf.keras.optimizers.Adam(learning_rate)

model = AE(original_dim)
# train(model, training_dataset, test_features, optimizer, epochs)
train_tensorboard(model, training_dataset, optimizer, epochs)                    

