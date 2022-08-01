# Task:
# postprocessing analysis
# use Joback method (using functional groups to calc. Cv) 
# find the rep. with qm9 and save the final file in a .csv file

import warnings
warnings.filterwarnings('ignore')
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
from rdkit import Chem
print ("!!!!!!!!!!!!!!!!!!!!!we are after importing rdkit!!!!!!!!!!!!!!!!!!")

from thermo.joback import Joback
import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# loading SMILES data using Chainer Chemistry
from chainer_chemistry.datasets.molnet import get_molnet_dataset
from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset
from chainer_chemistry.dataset.preprocessors import GGNNPreprocessor

# load the generated SMILES from the RCGAN Model
# if you want to test one sample
gen_SMILES = ['N#CC12C3CC1C2O3']
csv_name = './validsmiles_reinforce.csv'
gen_SMILES = pd.read_csv(csv_name)

jobacks = []
validated = []
valid_ids = []
for count, s in enumerate(gen_SMILES['gen_smiles'].values):
    try:
        J = Joback(s)
        jobacks.append(J.Cpig(298.15) * 0.2390057361)
        validated.append(s)
        valid_ids.append(count)     
    except:
        pass
print (jobacks)

# save the molecules that have Joback values.
val = {}
val['jobacks'] = jobacks
val['SMILES'] = validated
val = pd.DataFrame(val)
gen_SMILES2 = gen_SMILES.iloc[valid_ids, :]
gen_SMILES2['jobacks'] = jobacks
print (val.shape)
print (gen_SMILES2.shape)

# normalize Cv
min_cv = 21.02
max_cv = 42.302
gen_SMILES2['predcv_AE_latent'] = gen_SMILES2['predcv_AE_latent'] * (max_cv - min_cv) + min_cv
#val = pd.merge(val, gen_SMILES2, how = 'right', on = 'SMILES')
#print (val)
gen_SMILES2['Err_joback_pred'] = np.abs((gen_SMILES2['predcv_AE_latent'].values-
                                         gen_SMILES2['jobacks'].values)/
                                         gen_SMILES2['jobacks'].values)



# (Joback vs. predicted Cv by regressor)
mean_err = np.mean(np.abs((gen_SMILES2['predcv_AE_latent'].values -
                           gen_SMILES2['jobacks'].values) /
                           gen_SMILES2['jobacks'].values))
print ("mean error Joback(gen_SMILES) Vs. Predicted from regressor: ", mean_err)

gen_SMILES2.reset_index(drop = True, inplace = True)

csv_name = csv_name.replace('.csv', '')
gen_SMILES2.to_csv('{}_jobackadded.csv'.format(csv_name), index = False)


