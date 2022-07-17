# Strategy 1:
# Generate data after each epoch of training, if less than
# 10% error rate, and is a legit SMILES
# append to the real data
# Otherwise, append to fake data

# ADDING REINFORCEMENT MECHANISM
# Regenerate Normal sampling (define ranges), default: uniform

# IMPORTANT!!!!!!!!!!!!! DO NOT DROP DUPLICATE FOR RESULT .CSV

import warnings
warnings.filterwarnings('ignore')

import time
import os
import re
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import random
import numpy as np
from numpy import ndarray
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pickle

from tensorflow.keras.layers import (Input, Dropout, LSTM, Reshape, LeakyReLU,
                          Concatenate, ReLU, Flatten, Dense, Embedding,
                          BatchNormalization, Activation, SpatialDropout1D,
                          Conv2D, MaxPooling2D, UpSampling2D)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mse, binary_crossentropy
import tensorflow.keras.backend as K

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import np_utils

from tensorflow.keras.utils import  to_categorical
from IPython.display import clear_output
import matplotlib.pyplot as plt

from progressbar import ProgressBar
import seaborn as sns

from sklearn.metrics import r2_score


print ("!!!!!!!!! we are just before importing rdkit!!!!!")
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
from rdkit import Chem
print ("!!!!!!!!!!!!!!!!!!!!!we are after importing rdkit!!!!!!!!!!!!!!!!!!")

from scipy.stats import truncnorm
from sklearn.decomposition import PCA

import matplotlib.ticker as tk

import ntpath
import re

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score

from matplotlib.colors import ListedColormap

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
random.seed(12345)
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3667)
#session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, gpu_options=gpu_options)
#tf.set_random_seed(1234)
#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#K.set_session(sess)

with open('./../data/trainingsets/20000_train_regular_qm9/image_train.pickle', 'rb') as f:
    X_smiles_train, X_atoms_train, X_bonds_train, y_train = pickle.load(f)
    
with open('./../data/trainingsets/20000_train_regular_qm9/image_test.pickle', 'rb') as f:
    X_smiles_val, X_atoms_val, X_bonds_val, y_val = pickle.load(f)

with open('./../data/trainingsets/20000_train_regular_qm9/tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)
    
tokenizer[0] = ' '

# Subsampling has been done in the data preprocesses
"""
# Outlier removal
IQR = - np.quantile(y_train, 0.25) + np.quantile(y_train, 0.75)

lower_bound, upper_bound = np.quantile(y_train, 0.25) - 1.5 * IQR, np.quantile(y_train, 0.75) + 1.5 * IQR

idx = np.where((y_train >= lower_bound) & (y_train <= upper_bound))

y_train = y_train[idx]
X_smiles_train = X_smiles_train[idx]
X_atoms_train = X_atoms_train[idx]
X_bonds_train = X_bonds_train[idx]


# Outlier removal
IQR = - np.quantile(y_val, 0.25) + np.quantile(y_val, 0.75)

lower_bound, upper_bound = np.quantile(y_val, 0.25) - 1.5 * IQR, np.quantile(y_val, 0.75) + 1.5 * IQR

idx = np.where((y_val >= lower_bound) & (y_val <= upper_bound))

y_val = y_val[idx]
X_smiles_val = X_smiles_val[idx]
X_atoms_val = X_atoms_val[idx]
X_bonds_val = X_bonds_val[idx]
"""
# TRAIN/test split, we need 3 chuncks of samples, test, val, train
idx = np.random.choice(len(y_train), int(len(y_train) * 0.25), replace = False)
train_idx = np.setdiff1d(np.arange(len(y_train)), idx)

X_smiles_test, X_atoms_test, X_bonds_test, y_test = X_smiles_train[idx], X_atoms_train[idx], X_bonds_train[idx], y_train[idx]
X_smiles_train, X_atoms_train, X_bonds_train, y_train = X_smiles_train[train_idx], X_atoms_train[train_idx], X_bonds_train[train_idx], y_train[train_idx]


def norm(X: ndarray) -> ndarray:
    X = np.where(X == 0, -1.0, 1.0)
    return X

X_atoms_train, X_bonds_train = (norm(X_atoms_train),
                                norm(X_bonds_train))
X_atoms_val, X_bonds_val = (norm(X_atoms_val),
                            norm(X_bonds_val))

def y_norm(y: ndarray) -> ndarray:
    scaler_min = np.min(y)
    scaler_max = np.max(y)
    
    y = (y - scaler_min) / (scaler_max - scaler_min)
    
    return y, scaler_min, scaler_max

s_min1 = np.min (y_train)
s_max1 = np.max (y_train)

s_min2 = np.min(y_val)
s_max2 = np.max(y_val)
s_min = min(s_min1, s_min2)
s_max = max(s_max1, s_max2)

y_val = (y_val - s_min) / (s_max - s_min)
print ("min and max train data and test normalized", s_min, s_max, np.min(y_val), np.max(y_val))
# define s_min and s_max between 10-7s_max = 50
#s_min, s_max = 20, 50
y_train = (y_train - s_min) / (s_max - s_min)
print ("min and max train data and train normalized", s_min, s_max, np.min(y_train), np.max(y_train))

encoder = load_model('./../data/nns/encoder.h5')
decoder = load_model('./../data/nns/decoder.h5')

class Config:
    
    def __init__(self):
        self.Filters = [256, 128, 64]
        self.genFilters = [128, 128, 128]
        self.upFilters = [(2, 2), (2, 2), (2, 2)]
        
config = Config()

# Generator
z = Input(shape = (128, ))
y = Input(shape = (1, ))

h = Concatenate(axis = 1)([z, y])
h = Dense(1 * 1 * 128)(h)
R1 = Reshape([1, 1, 128])(h)
R2 = Reshape([1, 1, 128])(h)

for i in range(3):
    R1 = UpSampling2D(size = config.upFilters[i])(R1)
    C1 = Conv2D(filters = config.genFilters[i], 
               kernel_size = 2, 
               strides = 1, 
               padding = 'same')(R1)
    B1 = BatchNormalization()(C1)
    R1 = LeakyReLU(alpha = 0.2)(B1)

for i in range(3):
    R2 = UpSampling2D(size = config.upFilters[i])(R2)
    C2 = Conv2D(filters = config.genFilters[i], 
               kernel_size = 2, 
               strides = 1, 
               padding = 'same')(R2)
    B2 = BatchNormalization()(C2)
    R2 = LeakyReLU(alpha = 0.2)(B2)
    
R1 = Conv2D(1,
            kernel_size = 3,
            strides = 1,
            padding = 'valid',
            activation = 'tanh')(R1)
R2 = Conv2D(1,
            kernel_size = 3,
            strides = 1,
            padding = 'valid',
            activation = 'tanh')(R2)

generator = Model([z, y], [R1, R2])
print (generator.summary())
# Discriminator
#X = Input(shape = [6,6,2 ])
inp1 = Input(shape = [6, 6, 1])
inp2 = Input(shape = [6, 6, 1])

X1 = Concatenate()([inp1, inp2])
X = Flatten()(X1)
print (X.shape)
y2 = Concatenate(axis = 1)([X, y])
print (y2.shape)
for i in range(3):
		y2 = Dense(64, activation = 'relu')(y2)
		y2 = LeakyReLU(alpha = 0.2)(y2)
		y2 = Dropout(0.2)(y2)

O_dis = Dense(1, activation = 'sigmoid')(y2)


discriminator = Model([inp1, inp2, y], O_dis)
discriminator.compile(loss = 'binary_crossentropy', optimizer = Adam(lr = 5e-6, beta_1 = 0.5))
print (discriminator.summary()) 
# Regressor
inp1 = Input(shape = [6, 6, 1])
inp2 = Input(shape = [6, 6, 1])

yr = Concatenate()([inp1, inp2])

tower0 = Conv2D(32, 1, padding = 'same')(yr)
tower1 = Conv2D(64, 1, padding = 'same')(yr)
tower1 = Conv2D(64, 3, padding = 'same')(tower1)
tower2 = Conv2D(32, 1, padding = 'same')(yr)
tower2 = Conv2D(32, 5, padding = 'same')(tower2)
tower3 = MaxPooling2D(3, 1, padding = 'same')(yr)
tower3 = Conv2D(32, 1, padding = 'same')(tower3)
h = Concatenate()([tower0, tower1, tower2, tower3])
h = ReLU()(h)
h = MaxPooling2D(2, 1, padding = 'same')(h)

for i in range(6):
    tower0 = Conv2D(32, 1, padding = 'same')(h)
    tower1 = Conv2D(64, 1, padding = 'same')(h)
    tower1 = Conv2D(64, 3, padding = 'same')(tower1)
    tower2 = Conv2D(32, 1, padding = 'same')(h)
    tower2 = Conv2D(32, 5, padding = 'same')(tower2)
    tower3 = MaxPooling2D(3, 1, padding = 'same')(h)
    tower3 = Conv2D(32, 1, padding = 'same')(tower3)
    h = Concatenate()([tower0, tower1, tower2, tower3])
    h = ReLU()(h)
    if i % 2 == 0 and i != 0:
        h = MaxPooling2D(2, 1, padding = 'same')(h)
h = BatchNormalization()(h)

yr = Flatten()(h)
o = Dropout(0.2)(yr)
o = Dense(128)(o)

o_reg = Dropout(0.2)(o)
o_reg = Dense(1, activation = 'sigmoid')(o_reg)

regressor = Model([inp1, inp2], o_reg)
regressor_top = Model([inp1, inp2], o)

regressor.compile(loss = 'mse', optimizer = Adam(1e-5))
print (regressor.summary())
# combined
def build_combined(z, y,
                   regressor,
                   regressor_top,
                   discriminator):
    atoms_emb, bonds_emb = generator([z, y])

    
    discriminator.trainable = False
    regressor_top.trainable = False
    regressor.trainable = False

    y_pred = regressor([atoms_emb, bonds_emb])
    #latent = regressor_top([atoms_emb, bonds_emb])
    
    valid = discriminator([atoms_emb, bonds_emb, y])

    combined = Model([z, y], [valid, y_pred])

    combined.compile(loss = ['binary_crossentropy',
                             'mse'], 
                     loss_weights = [1.0, 25.0], 
                     optimizer = Adam(5e-6, beta_1 = 0.5))

    return combined

combined = build_combined(z, y,
                          regressor,
                          regressor_top,
                          discriminator)

train_atoms_embedding, train_bonds_embedding, _ = encoder.predict([X_atoms_train, X_bonds_train])

atoms_embedding, bonds_embedding, _ = encoder.predict([X_atoms_train, X_bonds_train])
atoms_val, bonds_val, _ = encoder.predict([X_atoms_val, X_bonds_val])

regressor = load_model('./../data/nns/regressor.h5')
regressor_top = load_model('./../data/nns/regressor_top.h5')

regressor.fit([atoms_embedding, bonds_embedding], 
              y_train,
              validation_data = ([atoms_val,
                                  bonds_val],
                                 y_val),
              batch_size = 32,
              epochs = 1,
              verbose = 1)

# Validating the regressor
#====#
atoms_embedding, bonds_embedding, _ = encoder.predict([X_atoms_train, X_bonds_train])
pred = regressor.predict([atoms_embedding, bonds_embedding])
print('Current R2 on Regressor for train data: {}'.format(r2_score(y_train, pred.reshape([-1]))))
print (pred)
print (y_train)
#====#
atoms_embedding, bonds_embedding, _ = encoder.predict([X_atoms_val, X_bonds_val])
pred = regressor.predict([atoms_embedding, bonds_embedding])
print('Current R2 on Regressor for validation data: {}'.format(r2_score(y_val, pred.reshape([-1]))))
print ("pred of validation data: ", pred )
print ("True validation values: ", y_val)
