from __future__ import absolute_import, division, print_function, unicode_literals

"""
Source: https://github.com/matiRLC/Keras-GAN/blob/master/gan/gan.py
Adapted by: matias@u.nus.edu
Updates:
    Try CGAN
    supervised discriminator
"""

# Tensorflow
import tensorflow as tf
from tensorflow.keras import layers
from IPython import display 
print(tf.__version__)

# Progress bar
from tqdm import tqdm

# Numpy, pandas, matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import PIL

# Sklearn
from sklearn.manifold import TSNE

class GAN():
    def __init__(self, data_dim):
        self.data_dim = data_dim # number of features
        self.latent_dim = data_dim # should be close in sizec with data_dim
         
        # optimizers
        self.generator_optimizer = tf.keras.optimizers.Adam(0.0002)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(0.0002)

        # Build the generator
        self.generator = self.build_generator()
        
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        
        # helper function to computer cross entropy loss
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def build_generator(self): # TODO: change architecture on constructor        
        model = tf.keras.Sequential()
        model.add(layers.Dense(150, use_bias=False, input_dim=self.latent_dim))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        
        model.add(layers.Dense(100, use_bias=False, input_dim=self.latent_dim))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        
        model.add(layers.Dense(100, use_bias=False, input_dim=self.latent_dim))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        
        model.add(layers.Dense(self.data_dim))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        assert model.output_shape == (None, 1)
        
        print("Generator Summary:")
        model.summary()

        return model

    def build_discriminator(self): # TODO: change architecture on constructor
        model = tf.keras.Sequential()
        model.add(layers.Dense(150, input_dim=self.data_dim))
        model.add(layers.LeakyReLU(alpha=0.2))
        
        model.add(layers.Dense(100))
        model.add(layers.LeakyReLU(alpha=0.2))
        
        model.add(layers.Dense(100))
        model.add(layers.LeakyReLU(alpha=0.2))
       
        model.add(layers.Dense(1))
        assert model.output_shape == (None, 1)
        
        print("Discriminator Summary:")
        model.summary()

        return model
    
    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    
    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)
    
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(self, data, BATCH_SIZE):
        noise = tf.random.normal([BATCH_SIZE, self.data_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_data = self.generator(noise, training=True)

            real_output = self.discriminator(data, training=True)
            fake_output = self.discriminator(generated_data, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, 
                                                   self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, 
                                                        self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, 
                                                self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, 
                                                    self.discriminator.trainable_variables))
    
    def train(self, dataset, EPOCHS, BATCH_SIZE=128, SAMPLE_INTERVAL=15):
        pbar = tqdm(total=EPOCHS) # progress bar
        
        for epoch in range(EPOCHS):
            for data_batch in dataset:
                self.train_step(data_batch, BATCH_SIZE)

            # Save the model every SAMPLE_INTERVAL epochs
            if epoch % SAMPLE_INTERVAL == 0:
                self.generate_data(epoch, BATCH_SIZE)
                display.clear_output(wait=True)
             #   checkpoint.save(file_prefix = checkpoint_prefix)

            pbar.update(1)
            
        # Generate after the final epoch
        display.clear_output(wait=True)
        self.generate_data(epoch, BATCH_SIZE * 5)
    
        pbar.close()
    
    def display_image(self, epoch_no):
        return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))
    
    def generate_data(self, epoch=1, BATCH_SIZE=128):
        noise = tf.random.normal([BATCH_SIZE * 5, self.latent_dim])
        
        generated_x = self.generator(noise, training=False)
        fig = plt.figure()
        plt.hist(generated_x.numpy(), bins=40, density=True, histtype='bar')
        plt.title("testing:" + str(epoch))
        plt.show()
#         fig.savefig("../output/vanillaGAN/d.png")
        