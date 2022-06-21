# Strategy 1:
# Generate data after each epoch of training, if less than
# 10% error rate, and is a legit SMILES
# append to the real data
# Otherwise, append to fake data

# ADDING REINFORCEMENT MECHANISM

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

from rdkit import Chem

print ("!!!!!!!!! we are just before importing rdkit!!!!!")
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
from rdkit import Chem
print ("!!!!!!!!!!!!!!!!!!!!!we are after importing rdkit!!!!!!!!!!!!!!!!!!")

from sklearn.decomposition import PCA


os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
random.seed(12345)
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3667)
#session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, gpu_options=gpu_options)
#tf.set_random_seed(1234)
#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#K.set_session(sess)

with open('image_train.pickle', 'rb') as f:
    X_smiles_train, X_atoms_train, X_bonds_train, y_train = pickle.load(f)
    
with open('image_test.pickle', 'rb') as f:
    X_smiles_val, X_atoms_val, X_bonds_val, y_val = pickle.load(f)

with open('tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)
    
tokenizer[0] = ' '

# Subsampling has been done in the data preprocesses

# Outlier removal
IQR = - np.quantile(y_train, 0.25) + np.quantile(y_train, 0.75)

lower_bound, upper_bound = np.quantile(y_train, 0.25) - 1.5 * IQR, np.quantile(y_train, 0.75) + 1.5 * IQR

idx = np.where((y_train >= lower_bound) & (y_train <= upper_bound))

y_train = y_train[idx]
X_smiles_train = X_smiles_train[idx]
X_atoms_train = X_atoms_train[idx]
X_bonds_train = X_bonds_train[idx]

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

y_train, s_min, s_max = y_norm(y_train)
y_val = (y_val - s_min) / (s_max - s_min)

encoder = load_model('encoder.h5')
decoder = load_model('decoder.h5')

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
X = Input(shape = (128, ))

y2 = Concatenate(axis = 1)([X, y])

for i in range(3):
    y2 = Dense(64, activation = 'relu')(y2)
    y2 = LeakyReLU(alpha = 0.2)(y2)
    y2 = Dropout(0.2)(y2)

O_dis = Dense(1, activation = 'sigmoid')(y2)

discriminator = Model([X, y], O_dis)
discriminator.compile(loss = 'binary_crossentropy', optimizer = Adam(lr = 5e-6, beta_1 = 0.5))

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

# combined
def build_combined(z, y,
                   regressor,
                   regressor_top,
                   discriminator):
    atoms_emb, bonds_emb = generator([z, y])

    discriminator.trainable = False
    regressor_top.trainable = False

    y_pred = regressor([atoms_emb, bonds_emb])
    latent = regressor_top([atoms_emb, bonds_emb])
    valid = discriminator([latent, y])

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

regressor = load_model('regressor.h5')
regressor_top = load_model('regressor_top.h5')
regressor.fit([atoms_embedding, bonds_embedding], 
              y_train,
              validation_data = ([atoms_val,
                                  bonds_val],
                                 y_val),
              batch_size = 32,
              epochs = 50,
              verbose = 1)

# Validating the regressor
#====#
atoms_embedding, bonds_embedding, _ = encoder.predict([X_atoms_train, X_bonds_train])
pred = regressor.predict([atoms_embedding, bonds_embedding])
print('Current R2 on Regressor for train data: {}'.format(r2_score(y_train, pred.reshape([-1]))))
#====#
atoms_embedding, bonds_embedding, _ = encoder.predict([X_atoms_val, X_bonds_val])
pred = regressor.predict([atoms_embedding, bonds_embedding])
print('Current R2 on Regressor for validation data: {}'.format(r2_score(y_val, pred.reshape([-1]))))

# Saving the currently trained models
#regressor.save('regressor.h5')
#regressor_top.save('regressor_top.h5')

#regressor = load_model('regressor.h5')
#regressor_top = load_model(regressor_top.h5')
generator = load_model ('generator.h5')
discriminator= load_model ('discriminator.h5')

regressor_top.trainable = False
regressor.trainable = False

epochs = 1
batch_size = 4
threshold = 0.3

reinforce_n = 10

batches = y_train.shape[0] // batch_size

G_Losses = []
D_Losses = []
R_Losses = []

for e in range(epochs):
    start = time.time()
    D_loss = []
    G_loss = []
    R_loss = []
    for b in range(batches):
        idx = np.arange(b * batch_size, (b + 1) * batch_size)
        # Subsample started for reinforcements
        idx = np.random.choice(idx, batch_size, replace = False)
        
        atoms_train = X_atoms_train[idx]
        bonds_train = X_bonds_train[idx]
        batch_y = y_train[idx]
        batch_z = np.random.normal(0, 1, size = (batch_size, 128))
        
        atoms_embedding, bonds_embedding, _ = encoder.predict([atoms_train, bonds_train])
        gen_atoms_embedding, gen_bonds_embedding = generator.predict([batch_z, batch_y])
        
        r_loss = regressor.train_on_batch([atoms_embedding, bonds_embedding], batch_y)
        R_loss.append(r_loss)
        
        real_latent = regressor_top.predict([atoms_embedding, bonds_embedding])
        fake_latent = regressor_top.predict([gen_atoms_embedding, gen_bonds_embedding])
        
        discriminator.trainable = True
        for _ in range(3):
            d_loss_real = discriminator.train_on_batch([real_latent, batch_y],
                                                       [0.9 * np.ones((batch_size, 1))])
            d_loss_fake = discriminator.train_on_batch([fake_latent, batch_y],
                                                       [np.zeros((batch_size, 1))])

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        D_loss.append(d_loss)
        
        discriminator.trainable = False
        g_loss = combined.train_on_batch([batch_z, batch_y], [0.9 * np.ones((batch_size, 1)), batch_y])
        G_loss.append(g_loss)
    
    D_Losses.append(np.mean(D_loss))
    G_Losses.append(np.mean(G_loss))
    R_Losses.append(np.mean(R_loss))
    
    print('====')
    print('Current epoch: {}/{}'.format((e + 1), epochs))
    print('D Loss: {}'.format(np.mean(D_loss)))
    print('G Loss: {}'.format(np.mean(G_loss)))
    print('R Loss: {}'.format(np.mean(R_loss)))
    print('====')
    print()
    
    # Reinforcement
    gen_error = []
    gen_smiles = []
    embeddings = []
    sample_ys = []
    for _ in range(1000):
        sample_y = np.random.uniform(s_min, s_max, size = [1,])
        sample_y = np.round(sample_y, 4)
        sample_y = (sample_y - s_min) / (s_max - s_min)
        sample_ys.append(sample_y)

        sample_z = np.random.normal(0, 1, size = (1, 128))

        sample_atoms_embedding, sample_bonds_embedding = generator.predict([sample_z, sample_y])
        embeddings.append((sample_atoms_embedding,
                           sample_bonds_embedding))
        reg_pred = regressor.predict([sample_atoms_embedding, sample_bonds_embedding])
        
        pred, desire = reg_pred[0][0], sample_y[0]
        gen_error.append(np.abs((pred - desire) / desire))

        dec_embedding = np.concatenate([sample_atoms_embedding, sample_bonds_embedding], axis = -1)
        smiles = decoder.predict(dec_embedding)[0]
        smiles = np.argmax(smiles, axis = 2).reshape([-1])

        c_smiles = ''
        for s in smiles:
            c_smiles += tokenizer[s]
        c_smiles = c_smiles.rstrip()
        gen_smiles.append(c_smiles)
        
    gen_error = np.asarray(gen_error)
        
    valid = 0
    idx_ = []
    for iter_, smiles in enumerate(gen_smiles):
        if ' ' in smiles[:-1]:
            continue
        m = Chem.MolFromSmiles(smiles[:-1],sanitize=False)
        if m is not None:
            valid += 1
            idx_.append(iter_)
    idx_ = np.asarray(idx_)
    
    if (e + 1) % 100 == 0:
        reinforce_n += 10
    
    # invalid smiles:
    fake_indices = np.setdiff1d(np.arange(1000), np.asarray(idx_))
    fake_indices = np.random.choice(fake_indices, reinforce_n * 5, replace = False)
    
    real_indices_ = np.intersect1d(np.where(gen_error < threshold)[0], idx_)
    sample_size = min(reinforce_n, len(real_indices_))
    real_indices = np.random.choice(real_indices_, sample_size, replace = False)
    
    if e >= 0:
        discriminator.trainable = True
        for real_index in real_indices:
            real_latent = regressor_top.predict([embeddings[real_index][0], embeddings[real_index][1]])
            _ = discriminator.train_on_batch([real_latent, sample_ys[real_index]],
                                             [0.9 * np.ones((1, 1))])

        for fake_index in fake_indices:
            fake_latent = regressor_top.predict([embeddings[fake_index][0], embeddings[fake_index][1]])
            _ = discriminator.train_on_batch([fake_latent, sample_ys[fake_index]],
                                             [np.zeros((1, 1))])
        discriminator.trainable = False
        
    # ==== #
    print('Currently valid SMILES: {}'.format(valid))
    print('Currently satisfying SMILES: {}'.format(len(real_indices_)))
    print('Currently unique satisfying generation: {}'.format(len(np.unique(np.array(gen_smiles)[real_indices_]))))
    print('Gen Sample is: {}, for {}'.format(c_smiles, sample_y))
    print('Predicted val: {}'.format(reg_pred))
    print('====')
    print()
    
    if (e + 1) % 10 == 0:
        plt.plot(G_Losses)
        plt.plot(D_Losses)
        plt.plot(R_Losses)
        plt.legend(['G Loss', 'D Loss', 'R Loss'])
        plt.show()
        plt.savefig("G_D_R_losses_{}.png".format (e+1))
    n_unique = len(np.unique(np.array(gen_smiles)[real_indices_]))
    n_valid = valid
    if valid > 450 and n_unique > 35:
        print('Criteria has satisified, training has ended')
        break
    end = time.time()
    print ("time for current epoch: ", (end - start))
with open('GAN_loss.pickle', 'wb') as f:
    pickle.dump((G_Losses, D_Losses, R_Losses), f)

# Saving the currently trained models
#regressor.save('regressor.h5')
#regressor_top.save('regressor_top.h5')
#generator.save('generator.h5')
#discriminator.save('discriminator.h5')

##====#

# Generation Study

#regressor = load_model('regressor.h5')
#regressor_top = load_model('regressor_top.h5')
#generator = load_model('generator.h5')
#discriminator = load_model('discriminator.h5')

encoder = load_model('encoder.h5')
decoder = load_model('decoder.h5')

# Generation workflow
# 1. Given a desired heat capacity
# 2. Generate 10,000 samples of SMILES embedding
# 3. Select the ones with small relative errors (< 10%)
# 4. Transfer them to SMILES
# 5. Filter out the invalid SMILES

# Generate 500 different values of heat capacities

from progressbar import ProgressBar

N = 50000
n_sample = 50

gen_error = []
gen_smiles = []
sample_ys = []
preds = []
gen_atoms_embedding = []
gen_bonds_embedding = []

pbar = ProgressBar()
for hc in pbar(range(n_sample)):
    try:
        sample_y = np.random.uniform(s_min, s_max, size = [1,])
        sample_y = np.round(sample_y, 3)
        sample_y = sample_y * np.ones([N,])

        sample_y_ = (sample_y - s_min) / (s_max - s_min)
        sample_z = np.random.normal(0, 1, size = (N, 128))

        sample_atoms_embedding, sample_bonds_embedding = generator.predict([sample_z, sample_y_])
        pred = regressor.predict([sample_atoms_embedding, sample_bonds_embedding]).reshape([-1])
        pred = pred * (s_max - s_min) + s_min

        gen_errors = np.abs((pred - sample_y) / sample_y).reshape([-1])

        accurate = np.where(gen_errors <= 0.1)[0]
        gen_errors = gen_errors[accurate]
        pred = pred[accurate]

        sample_y = sample_y[accurate]
        sample_atoms_embedding = sample_atoms_embedding[accurate]
        sample_bonds_embedding = sample_bonds_embedding[accurate]

        dec_embedding = np.concatenate([sample_atoms_embedding, sample_bonds_embedding], axis = -1)
        smiles = decoder.predict(dec_embedding)[0]
        smiles = np.argmax(smiles, axis = 2).reshape(smiles.shape[0], 35)

        generated_smiles = []
        for S in smiles:
            c_smiles = ''
            for s in S:
                c_smiles += tokenizer[s]
            c_smiles = c_smiles.rstrip()
            generated_smiles.append(c_smiles)
        generated_smiles = np.array(generated_smiles)

        all_gen_smiles = []
        idx = []
        for i, smiles in enumerate(generated_smiles):
            all_gen_smiles.append(smiles[:-1])

            if ' ' in smiles[:-1]:
                continue
            #m = Chem.MolFromSmiles(smiles[:-1],sanitize=False)
            m = Chem.MolFromSmiles(smiles[:-1])
            if m is not None:
                idx.append(i)

        idx = np.array(idx)
        all_gen_smiles = np.array(all_gen_smiles)

        gen_smiles.extend(list(all_gen_smiles[idx]))
        gen_error.extend(list(gen_errors[idx]))
        sample_ys.extend(list(sample_y[idx]))
        gen_atoms_embedding.extend(sample_atoms_embedding[idx])
        gen_bonds_embedding.extend(sample_bonds_embedding[idx])

        preds.extend(list(pred[idx]))
    except:
        print('Did not discover SMILES for HC: {}'.format(sample_y))
        pass
    
output = {}

for i, s in enumerate (gen_smiles):
    print (s)
    m = Chem.MolFromSmiles(s)
    ss = Chem.MolToSmiles (m)
    print (ss)
    gen_smiles[i] = ss

output['SMILES'] = gen_smiles
output['Desired hc'] = sample_ys
output['Predicted hc'] = preds
output['Error'] = gen_error

output = pd.DataFrame(output)
output = output.drop_duplicates(['SMILES'])

gen_atoms_embedding = np.array(gen_atoms_embedding)
gen_bonds_embedding = np.array(train_atoms_embedding)


####
# ANALYSIS
X_atoms_train_ = train_atoms_embedding.reshape([train_atoms_embedding.shape[0], 
                                        6 * 6])
X_bonds_train_ = train_bonds_embedding.reshape([train_bonds_embedding.shape[0], 
                                        6 * 6])

X_atoms_test_ = gen_atoms_embedding.reshape([gen_atoms_embedding.shape[0],
                                      6 * 6])
X_bonds_test_ = gen_bonds_embedding.reshape([gen_bonds_embedding.shape[0], 
                                      6 * 6])

pca_1 = PCA(n_components = 2)
X_atoms_train_ = pca_1.fit_transform(X_atoms_train_)
X_atoms_test_ = pca_1.transform(X_atoms_test_)

pca_2 = PCA(n_components = 2)
X_bonds_train_ = pca_2.fit_transform(X_bonds_train_)
X_bonds_test_ = pca_2.transform(X_bonds_test_)

# Atoms Distribution
plt.scatter(X_atoms_train_[:,0], X_atoms_train_[:,1], alpha = 0.3, c = 'blue');
plt.savefig("train_atom_dist.png")
plt.scatter(X_atoms_test_[:,0], X_atoms_test_[:,1], alpha = 0.3, c = 'red');
plt.savefig("test_atom_dist.png")
####

# Bonds Distribution
plt.scatter(X_bonds_train_[:,0], X_bonds_train_[:,1], alpha = 0.3, c = 'blue');
plt.savefig("train_bonds_dist.png")
plt.scatter(X_bonds_test_[:,0], X_bonds_test_[:,1], alpha = 0.3, c = 'red');
plt.savefig("test_bonds_dist.png")
# 31/500 failed (N = 10000)
# 2/50 failed (N = 50000)

output.reset_index(drop = True, inplace = True)

output.to_csv('generated_SMILES2.csv', index = False)

"""with open('gen_pickles.pickle', 'wb') as f:
    pickle.dump(gen_unique_pickles, f)
"""
