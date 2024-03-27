#%%
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
# %%
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# %%
train_df
# %%
train_df.info
# %%
train_df.shape
# %%
train_df.isna().sum()
# %%
train_df

# %%
X = train_df.drop(columns=['id','booking_status'])
y = train_df['booking_status']
# %%
X.shape
y.shape
y
# %%
# %%
from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val = train_test_split(X,y,shuffle=True,random_state=42,train_size=0.8)

# %%
print(X_train.shape)
print(y_train.shape)
# %%
from keras import layers

model = keras.Sequential([
    layers.BatchNormalization(input_shape=[X.shape[1]]),
    layers.Dense(256,activation='relu'),
    layers.BatchNormalization(),

    layers.Dense(256,activation='relu'),
    layers.BatchNormalization(),

    layers.Dense(128,activation='relu'),
    layers.BatchNormalization(),

    layers.Dense(1,activation='softmax')  
])
# %%
model.summary()
# %%
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    min_delta = 0.01,
    patience=15,
    restore_best_weights=True,
)

# %%
model.compile(
    loss='mse',
    metrics=['accuracy']
)
# %%
training = model.fit(
    X_train,y_train,
    validation_data=(X_val,y_val),
    batch_size=16,
    epochs=20,
    callbacks=[early_stopping]
)

history = pd.DataFrame(training.history)
history.loc[:, ['loss', 'val_loss','accuracy']].plot()
# %%
pred = model.predict(X_val)
# %%
pred.shape
# %%
