#%%
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers,callbacks
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# %%
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_sub = pd.read_csv('sample_submission.csv')
# %%
df_sub
# %%
df_train = df_train.drop(['id'],axis=1)
df_test = df_test.drop(['id'],axis=1)

# %%
df_train.shape
# %%
from autoviz.AutoViz_Class import AutoViz_Class

AV = AutoViz_Class()
AV.AutoViz(filename='',dfte=df_train,depVar='Strength',verbose=1,max_rows_analyzed=df_train.shape[0]
               ,max_cols_analyzed=df_train.shape[1])
# %%
X=df_train.iloc[:,:8]
y=df_train.Strength 
# %%
X
# %%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
Scale_test = scaler.fit_transform(df_test)

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y,shuffle=True,test_size=0.3)
# %%
tf.keras.activations.gelu
tf.keras.activations.selu
tf.keras.activations.elu

import keras
from keras import layers
model = tf.keras.Sequential([
    layers.Dense(128,activation='selu'),
    layers.Dropout(0.2),
    layers.Dense(64,activation='selu'),
    layers.Dropout(0.2),
    layers.Dense(32,activation='selu'),
    layers.Dropout(0.2),
    layers.Dense(16,activation='selu'),
    layers.Dropout(0.2),
    layers.Dense(8),
    layers.Dropout(0.2),
    layers.Dense(1,activation='linear')
])

# %%
early_stopping = callbacks.EarlyStopping(monitor="val_loss",patience=15,restore_best_weights=True)
# %%
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1 ,patience=5,min_lr=0.001)
# %%
y_test.shape
# %%
# To Comiple the model
model.compile(loss='mean_squared_error',
              
#otpimizer to take care on learning rate 
              
optimizer=tf.keras.optimizers.Adam(learning_rate=0.006)          
# to check on perfromance by means of metric
)
# %%
kuchbhiha = model.fit(X_train,y_train,epochs=8,callbacks=[early_stopping,reduce_lr],validation_data=(X_test, y_test))
# 在训练模型后，保存模型权重到一个HDF5文件
model.save_weights('model_weights.h5')

# %%
# To check on plot :(
pd.DataFrame(kuchbhiha.history).plot()
# %%
faltumodel= model.predict(Scale_test)
# %%
faltumodel.shape
# %%
# %%
