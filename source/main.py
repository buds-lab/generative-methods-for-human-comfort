#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:14:06 2019

@author: matias
Reading:
    https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html
    https://datasciencecampus.ons.gov.uk/projects/generative-adversarial-networks-gans-for-synthetic-dataset-generation-with-binary-classes/
    https://github.com/codyznash/GANs_for_Credit_Card_Data
    https://www.toptal.com/machine-learning/generative-adversarial-networks
    https://datasciencecampus.ons.gov.uk/projects/generative-adversarial-networks-gans-for-synthetic-dataset-generation-with-binary-classes/
    https://www.alexirpan.com/2017/02/22/wasserstein-gan.html
    https://medium.com/jungle-book/towards-data-set-augmentation-with-gans-9dd64e9628e6
    https://medium.com/gradientcrescent/generating-extinct-japanese-script-with-adversarial-autoencoders-theory-and-implementation-15f897d9ebbc
    https://towardsdatascience.com/deep-inside-autoencoders-7e41f319999f
    https://towardsdatascience.com/a-wizards-guide-to-adversarial-autoencoders-part-1-autoencoder-d9a5f8795af4
    
"""

# Pandas, numpy, and matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import MinMaxScaler

# User defined functions
from utils import *
from gan import *
#from vae import *
#from rbm import *

# External libraries
from tgan.model import TGANModel

# Load datasets
df = pd.read_csv("data/TCS_65_participants.csv")
df_band, df_survey = cleanTCSdataset(df)

# remove participants
del df_survey['Participant_No']

# update gpu list
gpu = checkGPU()

# -----------------------------------------------------------------------------
# vanilla GAN:
# https://www.toptal.com/machine-learning/generative-adversarial-networks
# -----------------------------------------------------------------------------

# filter for one class
df_survey_hot = df_survey[df_survey['Discrete Thermal Comfort_TA'] == 1]
del df_survey_hot['Discrete Thermal Comfort_TA']

# categorical to numbers
m = {'Male' : 0, 'Female' : 1}  
df_survey_hot['Gender'] = df_survey_hot['Gender'].map(m)

# model parameters
data_dim = df_survey_hot.shape[1] # num of features
data_cols = data_cols = [ i for i in df_survey_hot.columns]

rand_dim = data_dim # noise
base_n_count = 64 # neurons
nb_steps = 500 + 1 # Add one for logging of the last interval
batch_size = 64
k_d = 1  # number of critic/discriminator network updates per adversarial training step
k_g = 1  # number of generator network updates per adversarial training step
critic_pre_train_steps = 100 # 100  # number of steps to pre-train the critic before starting adversarial training
log_interval = 100 # 100  # interval (in steps) at which to log loss summaries and save plots of image samples to disc
learning_rate = 5e-4 # 5e-5
data_dir = 'cache/'

generator_model_path, discriminator_model_path, loss_pickle_path = None, None, None

show = True 

# scale
df_survey_hot_maxscaled = df_survey_hot.copy()
df_survey_hot_maxscaled[data_cols] /= df_survey_hot[data_cols].max()

arguments = [rand_dim, nb_steps, batch_size, 
             k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
            data_dir, generator_model_path, discriminator_model_path, loss_pickle_path, show ]

adversarial_training_GAN(arguments, df_survey_hot_maxscaled, data_cols) # GAN


model_steps = [ 0, 100, 200, 500, 1000, 2000, 5000]
rows = len(model_steps)
columns = 5

axarr = [[]]*len(model_steps)

fig = plt.figure(figsize=(14,rows*3))













# -----------------------------------------------------------------------------
# TGAN:
# https://github.com/DAI-Lab/TGAN
# -----------------------------------------------------------------------------

continuous_columns = [0,1,2,3,4,5,6]

# initialize model
tgan = TGANModel(
    continuous_columns,
    output='output',
    gpu=gpu,
    max_epoch=5,
    steps_per_epoch=10000,
    save_checkpoints=True,
    restore_session=True,
    batch_size=200,
    z_dim=200,
    noise=0.2,
    l2norm=0.00001,
    learning_rate=0.001,
    num_gen_rnn=100,
    num_gen_feature=100,
    num_dis_layers=1,
    num_dis_hidden=100,
    optimizer='AdamOptimizer',
)

# fit dataset
tgan.fit(df_survey)
model_path = 'models/tgan.pkl'
tgan.save(model_path)

# generate synth data
num_samples = 0.3 * len(df_survey)
samples = tgan.sample(num_samples)
samples.head(3)
samples.to_csv('data/synthdata.csv')

# Output labels (thermal comfort) histogram of synthdata
ax_output = samples['Discrete Thermal Comfort_TA'].value_counts().plot(
        kind = 'bar', figsize = (12,10))
ax_output.set_xlabel("Discrete Thermal Comfort_TA", fontsize = 15)
ax_output.set_ylabel("Count", fontsize = 15)
plt.show()

# -----------------------------------------------------------------------------
# TableGAN
# 
# -----------------------------------------------------------------------------


#X = np.asarray(test_df.iloc[:, :-1], dtype='float32')
#Y = np.asarray(test_df.iloc[:, -1], dtype='float32')
#X_scaled = MinMaxScaler().fit_transform(X) #0-1 scaling


#X_train, X_test, Y_train, Y_test = train_test_split(
#    X_scaled, Y, test_size=0.3, random_state=0)

# -----------------------------------------------------------------------------
# Generative model to sample synthetic data
# - RBM
# - Variational autoencoders
# - GAN
# - NB
# Validation:
#  Pearson correlation
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

# TODO: train on real data, test on test set of real data and synth data
# accuracies should be somewhat similar

# TODO: pearson correlation
# relations between the variables in the original data are preserved in the
# synthetic data

