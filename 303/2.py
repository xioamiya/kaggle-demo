#%%
import numpy as np
import pandas as pd
# %%
import matplotlib.pyplot as plt
# %%
train = pd.read_csv('train.csv')
train.shape
# %%
train = train.drop(['id'], axis=1)
# %%
train
# %%
train.hist(bins=40, figsize=(20,15),color ='c')
plt.show()
# %%
import seaborn as sns

df1 = train.select_dtypes(include='object')
df1
for i, col in enumerate(df1.columns):
    plt.figure(i)
    sns.countplot(x=col, data=df1,color ='c')
# %%
train.drop(['EmployeeCount','StandardHours','Over18'],axis=1,inplace=True)
# %%
features_num = train.select_dtypes(include=['int64','float64'])
features_num = features_num.drop(['Attrition'], axis=1)

features_cat = train.select_dtypes(include=['object'])

features_num = list(features_num.columns)
features_cat = list(features_cat.columns)
# %%
X = train.drop(['Attrition'], axis=1)
y = train[['Attrition']].values.ravel()
# %%
X.shape
# %%
y.shape
# %%
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.25, random_state=0,shuffle=True)
# %%
X_train.shape,y_train.shape,X_valid.shape
# %%
from sklearn.preprocessing import RobustScaler,OneHotEncoder
from sklearn.compose import make_column_transformer

preprocessor = make_column_transformer(
    (RobustScaler(), features_num),
    (OneHotEncoder(), features_cat),
)

X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)

X_train.shape
# %%
from tensorflow import keras
from keras import layers

model = keras.Sequential([
    layers.BatchNormalization(input_shape = [X_train.shape[1]]),
    layers.Dense(37,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(1,activation='sigmoid')
    ])
# %%
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)
# %%
model.summary()
# %%
early_stopping = keras.callbacks.EarlyStopping(
    patience=5,
    min_delta=0.0001,
    restore_best_weights=True,
)
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=64,
    epochs=200,
    callbacks=[early_stopping],
)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy")
history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot(title="Accuracy")
# %%
test = pd.read_csv('test.csv')
test.drop(['EmployeeCount','StandardHours','Over18'],axis=1,inplace=True)

X_test = preprocessor.fit_transform(test)

prediction = model.predict(X_test)

print(prediction)
# %%
prediction = pd.DataFrame(prediction)
prediction.columns = ['Attrition']
submissions = pd.DataFrame(pd.concat([test['id'],prediction],axis = 1))
submissions = submissions.reset_index(drop = True)

submissions.to_csv('submission.csv', index=False)
submissions.head()
# %%
