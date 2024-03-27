#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts
# %%
dftrain = pd.read_csv('train.csv')
dftest = pd.read_csv('test.csv')
# %%
print('-'*40)
print('Train Dataset Shape : {}'.format(dftrain.shape))
print('-'*40)
print('Test Dataset Shape : {}'.format(dftest.shape))
print('-'*40)
# %%
dftrain.info()
# %%
dftest.info()
# %%
dftrain=dftrain.drop(['id','cityCode'], axis=1)
dftest=dftest.drop(['id','cityCode'], axis=1)

dftrain.shape
# %%
features_train=dftrain.iloc[:,0:15]
sf=100000
label_train = dftrain.iloc[:,15]/sf
# %%
label_train
# %%
scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
# %%
dftest= scaler.transform(dftest)
# %%
train,val,label_train,label_val=tts(features_train,label_train,test_size=0.15,random_state=42)
# %%
import keras
from keras import layers

model = keras.Sequential([
    layers.Dense(128,activation='sigmoid'),
    layers.Dense(64,activation='sigmoid'),
    layers.Dense(32,activation='sigmoid'),

    layers.Dense(1,)  
])
# %%
# %%
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    min_delta=0.01,
    patience=15,
    restore_best_weights=True,
)
# %%
model.compile(
    loss='mae',
)
# %%
training = model.fit(
    train,label_train,
    validation_data=(val,label_val),
    batch_size=16,
    epochs=20,
    callbacks=[early_stopping]

)
history = pd.DataFrame(training.history)
history.loc[:, ['loss', 'val_loss']].plot()
# %%
