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
#Build the neural network
tf.random.set_seed(42)
#create a model
model=tf.keras.Sequential([
    tf.keras.layers.Dense(15,activation='sigmoid'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1)
])
#compile the model
model.compile(loss='mse',
              optimizer=tf.keras.optimizers.Adam(),
)
#fit the model
history=model.fit(train,label_train,validation_data=(val,label_val),epochs=250,batch_size=250,verbose=2)
# %%
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
val_predictions = model.predict(val)

# Assuming you've already trained your model and stored the validation predictions in val_predictions

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(label_val, val_predictions)
print(f"MAE: {mae}")

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(label_val, val_predictions))
print(f"RMSE: {rmse}")

# Calculate R-squared (R2)
r2 = r2_score(label_val, val_predictions)
print(f"R2: {r2}")

# %%
#plot history (Also known as loss  curve or a training curve)
pd.DataFrame(history.history).plot()
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.ylim([0,50])
# %%
price=np.round(model.predict(dftest),2)
pred=[]
for i in range(len(price)):
    pred.append(price[i][0]*sf)

sub = pd.read_csv('sample_submission.csv')
sub['price'] = pred
sub.to_csv('submission.csv' ,index = False)
# %%
