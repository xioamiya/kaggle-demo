#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %%
# Setup plotting
plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')

import warnings
warnings.filterwarnings('ignore')
# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from keras import layers
from sklearn.preprocessing import RobustScaler
# %%
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.head()
# %%
train.shape,test.shape
# %%
test.head()
# %%
rscale = RobustScaler()

train['Amount']=rscale.fit_transform(train['Amount'].values.reshape(-1,1))
train['Time']=rscale.fit_transform(train['Time'].values.reshape(-1,1))

test['Amount']=rscale.fit_transform(test['Amount'].values.reshape(-1,1))
test['Time']=rscale.fit_transform(test['Time'].values.reshape(-1,1))

# %%
train
# %%
legit = train[train['Class']==0]
legit
# %%
legit_sample = legit.sample(469)
legit_sample.shape
# %%
legit_sample.head()
# %%
fraud = train[train['Class']==1]
fraud.shape
# %%
fraud.head()
# %%
new_train = pd.concat([legit_sample,fraud],axis=0)
new_train.shape
# %%
new_train
# %%
X = new_train.drop('Class',axis=1)
y = new_train['Class']
X = new_train.drop('Class',axis=1)
y = new_train['Class']
# %%
X
# %%
[X.shape, y.shape]
# %%
X.head()
# %%
features_num = X.select_dtypes(exclude=['object']).copy()
features_num.columns
# %%
features_cat = X.select_dtypes(include=['object']).copy()
features_cat.columns
# %%
transformer_num = make_pipeline(
    SimpleImputer(strategy="constant"), # there are a few missing values
    StandardScaler(),
)
transformer_cat = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="NA"),
    OneHotEncoder(handle_unknown='ignore'),
)

preprocessor = make_column_transformer(
    (transformer_num, features_num),
    (transformer_cat, features_cat),
)

# stratify - make sure classes are evenlly represented across splits
X_train, X_valid, y_train, y_valid = \
    train_test_split(X, y, stratify=y, train_size=0.75)

# X_train = preprocessor.fit_transform(X_train)
# X_valid = preprocessor.transform(X_valid)

input_shape = [X_train.shape[1]]
# %%
[X_train.shape, y_train.shape]
# %%
[X_valid.shape, y_valid.shape]
# %%
X_train
# %%
import keras
model = keras.Sequential([
    layers.BatchNormalization(input_shape=input_shape),
    layers.Dropout(0.3),
    layers.Dense(1024, activation='relu'),    
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])
print(model.summary())
# %%
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)
# %%
early_stopping = keras.callbacks.EarlyStopping(
    patience=50,
    min_delta=0.001,
    restore_best_weights=True,
)

lr_schedule = keras.callbacks.ReduceLROnPlateau(
    patience=0,
    factor=0.2,
    min_lr=0.001,
)
# %%
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=1000,
    callbacks=[early_stopping, lr_schedule],
    verbose=1, # hide the output because we have so many epochs
)
# %%
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy")
history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot(title="Accuracy")

print(("Best Validation Loss: {:0.4f}" +\
      "\nBest Validation Accuracy: {:0.4f}")\
      .format(history_df['val_loss'].min(), 
              history_df['val_binary_accuracy'].max()))
# %%
