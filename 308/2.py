#%%
import numpy as np
import pandas as pd
# %%
import tensorflow as tf
X = pd.read_csv('train.csv')

X_test = pd.read_csv('test.csv')

# %%
y = X.price
y
# %%
X.drop(['price'],axis=1, inplace=True)

from sklearn.model_selection import train_test_split
X
# %%
X_train, X_valid, y_train, y_valid = train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=42)

# %%
X_train
# %%
from sklearn.preprocessing import OrdinalEncoder
object_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']
object_cols
# %%
ordinal_encoder = OrdinalEncoder()

X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])
X_test[object_cols] = ordinal_encoder.transform(X_test[object_cols])
# %%

X_train
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
X_train
# %%
y_train
# %%
training = model.fit(
    X_train,y_train,
    validation_data=(X_valid,y_valid),
    batch_size=16,
    epochs=20,
    callbacks=[early_stopping]

)
history = pd.DataFrame(training.history)
history.loc[:, ['loss', 'val_loss']].plot()
# %%
from sklearn.metrics import mean_squared_error

preds_valid =  model.predict(X_valid)
print(preds_valid[:10])
print(np.sqrt(mean_squared_error(preds_valid,y_valid)))
# %%
