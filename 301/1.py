#%%
import pandas as pd
import numpy as np
from hashlib import md5
# %%
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# %%
train.head()
# %%
train.drop(['MedInc','AveOccup','Latitude','Longitude'], inplace=True, axis=1)
test.drop(['MedInc','AveOccup','Latitude','Longitude'], inplace=True, axis=1)
# %%
x = train.drop("MedHouseVal",axis=1)
y = train['MedHouseVal']
# %%
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.2,random_state=42)
# %%
x_train
# %%
import keras
from keras import layers

model = keras.Sequential([
    layers.Dense(64,activation='relu'),

    layers.Dense(32,activation='relu'),

    layers.Dense(16,activation='relu'),

    layers.Dense(1,)  
])
# %%
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    min_delta=0.01,
    patience=15,
    restore_best_weights=True,
)
# %%
model.compile(
    loss='mse',
)
# %%
training = model.fit(
    x_train,y_train,
    validation_data=(x_val,y_val),
    batch_size=16,
    epochs=20,
    callbacks=[early_stopping]

)
history = pd.DataFrame(training.history)
history.loc[:, ['loss', 'val_loss']].plot()
# %%
from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(x_train,y_train)
# %%
predictions= clf.predict(x_val)
# %%
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor()
gbr.fit(x_train, y_train)
# %%
