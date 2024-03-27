#%%
from IPython.display import clear_output
clear_output()
# %%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_addons as tfa
from tabtransformertf.utils.preprocessing import df_to_dataset, build_categorical_prep
from tabtransformertf.models.fttransformer import FTTransformerEncoder, FTTransformer
# %%
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# %%
train_df.info()
# %%
CATEGORICAL_FEATURES = ['cut', 'color', 'clarity']
NUMERIC_FEATURES = ['carat', 'depth', 'table', 'x', 'y', 'z']
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET_FEATURE = 'price'
# %%
sc = StandardScaler()
sc.fit(train_df[NUMERIC_FEATURES])
train_df[NUMERIC_FEATURES]= sc.transform(train_df[NUMERIC_FEATURES])
train_df
# %%
train_data, val_data = train_test_split(train_df, test_size=0.2, shuffle=True, random_state=8)
# %%
train_data
# %%
# Transform to TF dataset
train_dataset = df_to_dataset(train_data[FEATURES + [TARGET_FEATURE]], TARGET_FEATURE, shuffle=True, batch_size=16)
val_dataset = df_to_dataset(val_data[FEATURES + [TARGET_FEATURE]], TARGET_FEATURE, shuffle=False, batch_size=16)
# %%
