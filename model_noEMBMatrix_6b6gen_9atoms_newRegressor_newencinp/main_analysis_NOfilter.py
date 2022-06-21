import warnings
warnings.filterwarnings('ignore')

print ("!!!!!!!!! we are just before importing rdkit!!!!!")
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

# load the generated smiles from the RCGAN Model
gen_SMILES = pd.read_csv('final_data_Feat_Cv_smiles.csv')

jobacks = []
validated = []
for s in gen_SMILES['SMILES'].values:
    try:
        J = Joback(s)
        jobacks.append(J.Cpig(298.15) * 0.2390057361)
        validated.append(s)
    except:
        pass

print ("length of validated smiles by Joback {} Vs. total gen_smiles {}".\
        format (len(validated), len (gen_SMILES['SMILES'])))

val = {}
val['jobacks'] = jobacks
val['SMILES'] = validated
val = pd.DataFrame(val)
print (val)
print (gen_SMILES)
val = pd.merge(val, gen_SMILES, how = 'left', on = 'SMILES')
print (val)

# error using Joback method mean and median 
# (joback vs. Desired Cv)
mean_err = np.mean(np.abs((val['Desired hc'].values - 
                           val['jobacks'].values) / 
                           val['jobacks'].values)
)
print ("mean error Joback(gen_SMILES) Vs. Sampled_Desired: ", mean_err)
median_err = np.median(np.abs((val['Desired hc'].values - 
                               val['jobacks'].values) / 
                               val['jobacks'].values)
)
print ("median error Joback(gen_SMILES) Vs. Sampled_Desired: ", median_err)

# error using Joback method mean and median 
# (Joback vs. predicted Cv by regressor)
mean_err = np.mean(np.abs((val['Predicted hc'].values - 
                           val['jobacks'].values) / 
                           val['jobacks'].values)
)
print ("mean error Joback(gen_SMILES) Vs. Predicted from regressor: ", mean_err)
median_err = np.median(np.abs((val['Predicted hc'].values - 
                               val['jobacks'].values) /  
                               val['jobacks'].values)
)
print ("median error Joback(gen_SMILES) Vs. Predicted from regressor: ", median_err)

# find the best candidates in generated smiles (criteria: <0.05)
val_accurate = pd.DataFrame({'SMILES': [],
                             'Desired hc': [],
                             'Predicted hc': [],
                             'jobacks': []})
accurate = []
print (val_accurate)
"""
for i, s in enumerate (val['SMILES'].values):
    if (np.abs((val['Desired hc'].values[i] - val['jobacks'].values[i]) / val['jobacks'].values[i]) < 0.07 and
        np.abs((val['Desired hc'].values[i] - val['jobacks'].values[i]) / val['jobacks'].values[i]) > 0.03 ) :
        accurate.append(i)
print (accurate)
"""

for i, s in enumerate (val['SMILES'].values):
    if np.abs((val['Desired hc'].values[i] - val['jobacks'].values[i]) / val['jobacks'].values[i]) < 0.04:
        accurate.append(i)
print (accurate)

for ii, a in enumerate (accurate):
    print (" i and a from accurate",ii, a)
    val_accurate.loc[ii,:] = val.iloc[a,:]
print (val_accurate)

sort_val_accurate = val_accurate.sort_values ('Desired hc')
print (sort_val_accurate) 

# accuracy of the the model Joback vs. predicted and desired Cv (accurate < 5%)
mean_err = np.mean(np.abs((val_accurate['Predicted hc'].values - 
                           val_accurate['jobacks'].values) / 
                           val_accurate['jobacks'].values)
)
print ("mean error Joback(gen_SMILES) Vs.Predicted from regressor (for accurate Cv(<5%): ", 
        mean_err)

median_err = np.median(np.abs((val_accurate['Predicted hc'].values - 
                               val_accurate['jobacks'].values) / 
                               val_accurate['jobacks'].values)
)
print ("median error Joback(gen_SMILES) Vs.Predicted from regressor(for accurate Cv(<5%) : ", 
        median_err)

mean_err = np.mean(np.abs((val_accurate['Desired hc'].values -
                           val_accurate['jobacks'].values) /
                           val_accurate['jobacks'].values)
)
print ("mean error Joback(gen_SMILES) Vs.Desired from regressor (for accurate Cv(<5%): ",
        mean_err)

median_err = np.median(np.abs((val_accurate['Desired hc'].values -
                               val_accurate['jobacks'].values) /
                               val_accurate['jobacks'].values)
)
print ("median error Joback(gen_SMILES) Vs.Desired from regressor (for accurate Cv(<5%) : ",
        median_err)

num_acc_l0p1 = np.sum(np.abs((val['Desired hc'].values - 
                               val['jobacks'].values) / 
                               val['jobacks'].values) < 0.1)

plt.scatter(val['Desired hc'].values, val['jobacks'].values)
plt.savefig("Desired_VS_joback.png")

plt.clf()
#sns.distplot(np.abs((val['Desired hc'].values - val['jobacks'].values) / val['jobacks'].values))
sns.distplot(((val['Desired hc'].values - val['jobacks'].values) / val['jobacks'].values))

plt.savefig("err_Distr_DesiVSJobsck.png")

#1131/3020
num_acc_l0p025 = np.sum(gen_SMILES['Error'].values < 0.025, dtype = np.int32)
num_acc_0p025_0p05 =  np.sum((gen_SMILES['Error'].values >= 0.025) & (gen_SMILES['Error'].values < 0.05), dtype = np.int32)
num_acc_g0p05 =  np.sum(gen_SMILES['Error'].values > 0.05, dtype = np.int32)
total = num_acc_l0p025 + num_acc_0p025_0p05 + num_acc_g0p05

print ("type of accurate l0p025: ", type (num_acc_l0p025))
print (num_acc_l0p025)

summ = float(gen_SMILES.shape[0])
print ("total SMILES is : {}".format(gen_SMILES.shape))
plt.figure(figsize = (5, 5))

plt.bar(['< 2.5%', '2.5% - 5%', '5% - 10%'],
        [num_acc_l0p025 / total, num_acc_0p025_0p05 / total,num_acc_g0p05 / total],
        color = ['green', 'blue', 'red'],
        alpha = 0.7)
plt.grid()
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.savefig("error_Distrib_5_2p5percent.png")

val_accurate.reset_index(drop = True, inplace = True)

val_accurate.to_csv('gen_SMILES_accur_joback.csv', index = False)

preprocessor = GGNNPreprocessor()
data = get_molnet_dataset('qm9',
                          labels = 'cv',
                          preprocessor = preprocessor,
                          return_smiles = True,
                          frac_train = 1.0,
                          frac_valid = 0.0,
                          frac_test = 0.0
                         )

smiles = data['smiles'][0]

smiles = smiles.astype('str')

data_smiles = []
data_cv = []
for i, s in enumerate(smiles):
    data_smiles.append(s)
    data_cv.append(data['dataset'][0][i][2][0])

jobacks = []
validated = []
for i, s in enumerate(data_smiles):
    try:
        J = Joback(s)
        jobacks.append(J.Cpig(298.15))
        validated.append(i)
    except:
        pass

data_cv = np.array(data_cv)[validated]
data_smiles = np.array(data_smiles)

jobacks = np.array(jobacks)

qm9_joback_Mean_relerr = np.mean(np.abs((data_cv - jobacks * 0.2390057361) / (jobacks * 0.2390057361)))
# 6.0% difference between qm9 cv and joback's cv
print ("qm9_joback_Mean_relerris :{}".format(qm9_joback_Mean_relerr))
gen_unique = gen_SMILES['SMILES'].values

data = {}
data['SMILES'] = data_smiles[validated]
data['cv'] = data_cv
data = pd.DataFrame(data)

print (data)

database_samples = pd.merge(gen_SMILES, data, on = 'SMILES', how = 'inner')

print ( "same generated smiles compared to qm9 lib is:{}".format(database_samples))

database_samples_accurate = pd.merge(val_accurate, data, on = 'SMILES', how = 'inner')

print ( "same generated accurate smiles compared to qm9 lib is:{}".format(database_samples_accurate))
rep_index = []
for i, smil in enumerate (database_samples_accurate['SMILES'].values):
    for j, smi in enumerate (val_accurate ['SMILES'].values):
         if (smi == smil):
             print ("smi ==  smil")
             rep_index.append(j)
             #val_accurate = val_accurate.drop([j])
print (rep_index)
val_accurate = val_accurate.drop(rep_index)

val_accurate.reset_index(drop = True, inplace = True)

val_accurate.to_csv('gen_SMILES_accurjoback_qm9reprem.csv', index = False)

mean_rel_diff_desired_cvqm9 = np.mean(np.abs((database_samples['Desired hc'].values -
                                              database_samples['cv'].values) / 
                                              database_samples['cv'].values))
print ("mean of rel diff BW Desired (sampled in design model) and cv from qm9: {}".
                                                 format(mean_rel_diff_desired_cvqm9))

median_rel_diff_desired_cvqm9 = np.median(np.abs((database_samples['Desired hc'].values
                       - database_samples['cv'].values) / database_samples['cv'].values))
print ("median of rel diff BW Desired (sampled in design model) and cv from qm9: {}".
                                                 format(median_rel_diff_desired_cvqm9))

mean_rel_diff_desired_cvqm9 = np.mean(np.abs((database_samples['Predicted hc'].values
                       - database_samples['cv'].values) / database_samples['cv'].values))
print ("mean of rel diff BW Predicted (from regresor) and cv from qm9: {}".
                                                 format(mean_rel_diff_desired_cvqm9))

median_rel_diff_desired_cvqm9 = np.median(np.abs((database_samples['Predicted hc'].values
                       - database_samples['cv'].values) / database_samples['cv'].values))
print ("median of rel diff BW Predicted (from regressor) and cv from qm9: {}".
                                                 format(median_rel_diff_desired_cvqm9))

"""
sns.distplot(np.abs((database_samples['Desired hc'].values - database_samples['cv'].values) / database_samples['cv'].values))

ins = 0

for G in gen_unique:
    # smile = G[:-1]
    if G in smiles:
        ins += 1

ins, len(gen_unique), ins / len(gen_unique)

smiles = smiles.astype('str')
gen_unique = gen_unique.astype('str')

print ("len (smiles): {} , len (gen_unique): {}".format(len(smiles), len(gen_unique)))

X_smiles = np.concatenate((smiles, gen_unique))

MAX_NB_WORDS = 23
MAX_SEQUENCE_LENGTH = 35

tokenizer = Tokenizer(num_words = MAX_NB_WORDS,
                      char_level = True,
                      filters = '',
                      lower = False)
tokenizer.fit_on_texts(X_smiles)

X_smiles = tokenizer.texts_to_sequences(X_smiles)
X_smiles = pad_sequences(X_smiles,
                         maxlen = MAX_SEQUENCE_LENGTH,
                         padding = 'post')

X_smiles = to_categorical(X_smiles)

X_train = X_smiles[:-2710]
X_gen = X_smiles[-2710:]

X_train = X_train.reshape([X_train.shape[0], 35 * 23])
X_gen = X_gen.reshape([X_gen.shape[0], 35 * 23])

pca = PCA(n_components = 2)
X_train_ = pca.fit_transform(X_train)
X_gen_ = pca.transform(X_gen)

# Atoms Distribution
plt.scatter(X_train_[:,0], X_train_[:,1], alpha = 0.3, c = 'blue');
plt.scatter(X_gen_[:,0], X_gen_[:,1], alpha = 0.3, c = 'red');
plt.grid()
plt.savefig ("atoms_Distri.png")
"""
