#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

submission = pd.read_csv('sample_submission.csv')

# %%
train
# %%
train.isna().sum()
# %%
sns.displot(train['cost'], kde=True)
# %%
train['cost'].value_counts()
# %%
X = train.iloc[:, :-1]
Y = train.iloc[:, -1]

X = X.drop('id',axis=1)
X.shape
# %%
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping


print(tf.__version__)
# %%
X.describe().transpose()[['mean', 'std']]
# %%
#normalization
normalizer = tf.keras.layers.Normalization(axis=1)
normalizer.adapt(np.array(X))
print(normalizer.mean.numpy())
# %%
first = np.array(X[:1])

with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())
# %%
def build_and_compile_model(norm):
    model = keras.Sequential([
       norm,
       layers.Dense(64, activation='relu'),
       layers.Dropout(0.2),
       layers.Dense(64,activation='relu'),
       layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',optimizer=tf.keras.optimizers.Adam(0.001))
    return model


# %%
dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()
# %%
early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=25, verbose=1)
mc = ModelCheckpoint ('best_model.h5', monitor='val_loss', mode='min', save_best_only=True) 
history = dnn_model.fit(X,Y,validation_split = 0.2,verbose=1, epochs=20, callbacks=[early_stopping, mc])
# %%
test.shape
# %%
import h5py

# 打开 .h5 文件
file = h5py.File('best_model.h5', 'r')

# 读取数据
# 例如，如果你想读取一个名为 'weights' 的数据集：

# 或者读取一个名为 'group/dataset' 的数据集：

file
# 关闭文件

# %%
