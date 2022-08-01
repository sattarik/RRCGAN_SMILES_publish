# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jeBcxYNVbeeknPCjl9IAp8ZOp07vtRqu
"""

import warnings
warnings.filterwarnings('ignore')

import os
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
                          Conv2D, MaxPooling2D, Softmax, 
                           Lambda)
#from tensorflow.keras.layers.experimental.preprocessing import CategoryEncoding
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam, RMSprop
import tensorflow.keras.backend as K
from tensorflow.keras.activations import tanh

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#import np_utils
from tensorflow.keras.utils import to_categorical
from IPython.display import clear_output
import matplotlib.pyplot as plt
import csv
from progressbar import ProgressBar
import seaborn as sns
from tensorflow import random as randomtf
from tensorflow.keras.backend import argmax as argmax

from tensorflow import one_hot

randomtf.set_seed(1)
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
random.seed(12345)
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
#session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, gpu_options=gpu_options)
#tf.set_random_seed(1234)
#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#K.set_session(sess)

print ('tf version', tf.__version__)

# Commented out IPython magic to ensure Python compatibility.
# %cd ./drive/MyDrive/RCGAN_SMILES/RCGAN_SMILES/

with open('./../data/trainingsets/60000_train_regular_qm9/image_train.pickle', 'rb') as f:
    X_smiles_train, SMILES_train, X_atoms_train, X_bonds_train, y_train = pickle.load(f)

with open('./../data/trainingsets/60000_train_regular_qm9/image_test.pickle', 'rb') as f:
    X_smiles_test, SMILES_test, X_atoms_test, X_bonds_test, y_test = pickle.load(f)

# Outlier removal train
IQR = - np.quantile(y_train, 0.25) + np.quantile(y_train, 0.75)

lower_bound, upper_bound = np.quantile(y_train, 0.25) - 1.5 * IQR, np.quantile(y_train, 0.75) + 1.5 * IQR

idx = np.where((y_train >= lower_bound) & (y_train <= upper_bound))

y_train = y_train[idx]
X_smiles_train = X_smiles_train[idx]
X_atoms_train = X_atoms_train[idx]
X_bonds_train = X_bonds_train[idx]
SMILES_train = SMILES_train [idx]
# Subsampling has been done in the data preprocesses
print ('X_smiles_train shape: ', X_smiles_train.shape)
print ('X_smiles_test shape: ', X_smiles_test.shape)
print ('X_smiles_train first sample: ', X_smiles_train[0][:][:][:])
print ('First SMILES train: ', SMILES_train[0])
print ('first cv', y_train[0])
# Outlier removal test
IQR = - np.quantile(y_test, 0.25) + np.quantile(y_test, 0.75)

lower_bound, upper_bound = np.quantile(y_test, 0.25) - 1.5 * IQR, np.quantile(y_test, 0.75) + 1.5 * IQR

idx = np.where((y_test >= lower_bound) & (y_test <= upper_bound))

y_test = y_test[idx]
X_smiles_test = X_smiles_test[idx]
X_atoms_test = X_atoms_test[idx]
X_bonds_test = X_bonds_test[idx]
SMILES_test = SMILES_test [idx]
def norm(X: ndarray) -> ndarray:
    X = np.where(X == 0, -1.0, 1.0)
    return X

X_atoms_train, X_bonds_train = (norm(X_atoms_train),
                                norm(X_bonds_train))

X_atoms_test, X_bonds_test = (norm(X_atoms_test),
                              norm(X_bonds_test))

X_smiles_train.shape

# Normalize y_train, y_test
s_min1 = np.min (y_train)
s_max1 = np.max (y_train)

s_min2 = np.min(y_test)
s_max2 = np.max(y_test)

# Use these values for generation.
s_min_dataset = min(s_min1, s_min2)
s_max_dataset = max(s_max1, s_max2)
s_min_norm, s_max_norm = s_min_dataset, 55

y_train = (y_train - s_min_norm) / (s_max_norm - s_min_norm)
y_test = (y_test - s_min_norm) / (s_max_norm -   s_min_norm)

print ("min and max used to normalized", s_min_norm, s_max_norm) 
print ("min and max of this data testing after norm", np.min(y_test), np.max(y_test))
print ("min and max of this data training after norm", np.min(y_train), np.max(y_train))
print ("min and max of dataset to be used for generation", s_min_dataset, s_max_dataset)

# Encoding to an image embedding

# ENCODER
#inp_1 = Input(shape = [9, 10, 1])
#inp_2 = Input(shape = [9, 9, 4])
inp_1 = Input(shape = [35, 23, 1])

y1 = Conv2D(64, (15, 3), strides = 1, padding = 'valid')(inp_1)
y1 = LeakyReLU(alpha = 0.2)(y1)
y1 = BatchNormalization()(y1)

y1 = Conv2D(64, 6, strides = 1, padding = 'valid')(y1)
y1 = LeakyReLU(alpha = 0.2)(y1)
y1 = BatchNormalization()(y1)

y1 = Conv2D(64, 6, strides = 1, padding = 'valid')(y1)
y1 = LeakyReLU(alpha = 0.2)(y1)
y1 = BatchNormalization()(y1)

y1 = Conv2D(64, 6, strides = 1, padding = 'valid')(y1)
y1 = LeakyReLU(alpha = 0.2)(y1)
y1 = BatchNormalization()(y1)

y1_emb = Conv2D(1, 3, strides = 1, padding = 'same',
            activation = 'tanh')(y1)

y2 = Conv2D(64, (15, 3), strides = 1, padding = 'valid')(inp_1)
y2 = LeakyReLU(alpha = 0.2)(y2)
y2 = BatchNormalization()(y2)

y2 = Conv2D(64, 6, strides = 1, padding = 'valid')(y2)
y2 = LeakyReLU(alpha = 0.2)(y2)
y2 = BatchNormalization()(y2)

y2 = Conv2D(64, 6, strides = 1, padding = 'valid')(y2)
y2 = LeakyReLU(alpha = 0.2)(y2)
y2 = BatchNormalization()(y2)

y2 = Conv2D(64, 6, strides = 1, padding = 'valid')(y2)
y2 = LeakyReLU(alpha = 0.2)(y2)
y2 = BatchNormalization()(y2)

y2_emb = Conv2D(1, 3, strides = 1, padding = 'same',
                activation = 'tanh')(y2)

####
y_out = Concatenate()([y1_emb, y2_emb])

# DECODER
emb_in = Input(shape = [6, 6, 2])

tower0 = Conv2D(32, 1, padding = 'same')(emb_in)
tower1 = Conv2D(64, 1, padding = 'same')(emb_in)
tower1 = Conv2D(64, 3, padding = 'same')(tower1)
tower2 = Conv2D(32, 1, padding = 'same')(emb_in)
tower2 = Conv2D(32, 5, padding = 'same')(tower2)
tower3 = MaxPooling2D(3, 1, padding = 'same')(emb_in)
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

y = Flatten()(h)

y = Dense(256, activation = 'relu')(y)
y_cv = Dense(64, activation = 'relu')(y)
y = Dropout(0.2)(y)
y = Dense(128, activation = 'relu')(y)
y = Dropout(0.2)(y)
y = Dense(128, activation = 'relu')(y)
y = Dropout(0.2)(y)
y = Dense(35 * 23)(y)
y = Reshape([35, 23, 1])(y)
y = Softmax(axis = 2)(y)


y_cv = Dropout(0.2)(y_cv)
y_cv = Dense(128, activation = 'relu')(y_cv)
y_cv = Dropout(0.2)(y_cv)
y_cv = Dense(128, activation = 'relu')(y_cv)
y_cv = Dense(1, activation = 'sigmoid')(y_cv)

encoder = Model([inp_1], [y1_emb, y2_emb, y_out], name = 'Encoder')
decoder = Model(emb_in, [y, y_cv], name = 'Decoder')
#print (encoder.summary())
#print (decoder.summary())
outputs = decoder(encoder([inp_1])[2])
#output_2 = decoder(encoder([inp_1])[2])[0]
#output_2 = argmax (output_2, axis=2)
#print (output_2)
#output_2 = Reshape([1, 35])(output_2)

#print (output_2)
# Use IntegerLookup to build an index of the feature values and encode output.
#lookup = IntegerLookup(output_mode="one_hot")
#lookup.adapt(data)
# Convert new test data (which includes unknown feature values)
#outputs = lookup(outputs)
#output_2 = one_hot(output_2, depth=23)


model = Model(inp_1, outputs, name = 'ae')
print (model.summary())

"""
lr_schedule = keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10 ** (epoch / 20)
)

encoder = load_model('encoder.h5')
decoder = load_model('decoder.h5')
model = load_model('ae_model.h5')

model.compile(optimizer = Adam(learning_rate = 1e-8),
              loss = ['binary_crossentropy', 'mse'])
history = model.fit([X_atoms_train, X_bonds_train],
                    [X_smiles_train, y_train],
                    validation_data = ([X_atoms_test, X_bonds_test],
                                       [X_smiles_test, y_test]),
                    epochs = 1,
                    batch_size = 32,
                    verbose = 1,
                    callbacks = [lr_schedule])



plt.semilogx(history.history['lr'],
             history.history['val_Decoder_loss'])

encoder = Model([inp_1, inp_2], [y1_emb, y2_emb, y_out], name = 'Encoder')
decoder = Model(emb_in, [y, y_cv], name = 'Decoder')

print (encoder.summary())
print (decoder.summary())

outputs = decoder(encoder([inp_1, inp_2])[2])
model = Model([inp_1, inp_2], outputs, name = 'ae')

model.compile(optimizer = Adam(learning_rate = 9e-5),
              loss = ['binary_crossentropy', 'mse'])

model.fit([X_atoms_train, X_bonds_train],
                    [X_smiles_train, y_train],
                    validation_data = ([X_atoms_test, X_bonds_test],
                                       [X_smiles_test, y_test]),
                    epochs = 1,
                    batch_size = 32,
                    verbose = 1)
"""

try:
    encoder = load_model('./../data/nns_9HA_noemb_6b6/encoder_newencinp.h5')
    decoder = load_model('./../data/nns_9HA_noemb_6b6/decoder_newencinp.h5')
    model = load_model  ('./../data/nns_9HA_noemb_6b6/ae_model_newencinp.h5')
    print (".h5 files were read")
except:
    print ("NO .h5 trained files")
    model.compile(optimizer = Adam(learning_rate = 9e-5),
              loss = ['binary_crossentropy', 'mse'])
    pass


"""
model.compile(optimizer = Adam(learning_rate = 9e-5),
              loss = ['binary_crossentropy', 'mse'])
"""
history = model.fit(X_smiles_train,
                    [X_smiles_train, y_train],
                    validation_data = (X_smiles_test,
                                       [X_smiles_test, y_test]),
                    epochs = 10,
                    batch_size = 32,
                    verbose = 1)

print(history.history.keys())
# summarize history for loss
plt.close()
plt.plot(history.history['val_loss'])
plt.title('Autoencoder loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig("autoencoderloss9HA_400_600.png", dpi=300)



model.save  ('./../data/nns_9HA_noemb_6b6/ae_model_newencinp_trans.h5')
encoder.save('./../data/nns_9HA_noemb_6b6/encoder_newencinp_trans.h5')
decoder.save('./../data/nns_9HA_noemb_6b6/decoder_newencinp_trans.h5')



for i in [5, 10, 32, 88, 99]:
    plt.subplot(121)
    plt.imshow(X_smiles_train[i].reshape([35, 23]))
    test_sample_pred = decoder.predict(encoder.predict([X_smiles_train[i:(i+2)]])[2])[0][0]
    plt.subplot(122)
    plt.imshow(test_sample_pred.reshape([35, 23]))
    plt.show()
    plt.savefig("smiles_{}_train.png".format(i)) 

# get i and i+2 to have (2,9,10,1) shape
# if only i was chosen, the should be (9,10,1)
output = decoder.predict(encoder.predict([X_smiles_train[0:2][:][:][:]])[2])[0][0]
output = argmax (output, axis=1)
output = to_categorical (output, num_classes = 23)
print (SMILES_train[0])
print (output.shape)
print ('output of decoder', output)
print (y_train[0])

for i in [5, 10, 32, 88, 99]:
    plt.subplot(121)
    plt.imshow(X_smiles_test[i].reshape([35, 23]))
    test_sample_pred = decoder.predict(encoder.predict([X_smiles_test[i:(i+2)]])[2])[0][0]
    plt.subplot(122)
    plt.imshow(test_sample_pred.reshape([35, 23]))
    plt.show()
    plt.savefig("smiles_{}_test.png".format(i))


