#%%
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from xgboost import XGBClassifier
import catboost
from catboost import CatBoostClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
# %%
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# %%
train_df.head()
# %%
train_df.info()
# %%
train_df.isna().sum()
# %%
data_types = train_df.dtypes
data_types
# %%
object_columns = data_types[data_types == 'object'].index
number_columns = data_types[data_types == 'number'].index
# %%
object_columns
# %%
from pandas import get_dummies
train_df = pd.get_dummies(train_df,columns=object_columns)
# %%
train_df
# %%
test_df
# %%
object_columns = test_df.dtypes[data_types == 'object'].index
number_columns = test_df.dtypes[data_types == 'number'].index
# %%
test_df = pd.get_dummies(test_df,columns=object_columns)
# %%
train_df
# %%
train_df = train_df.drop('id',axis=1)
# %%
train_df
# %%
value_counts = train_df['Attrition'].value_counts()
# %%
value_counts
# %%
sns.set(style="whitegrid")  # 设置样式
plt.figure(figsize=(10, 6))  # 设置图形大小
sns.countplot(data=train_df, x='Attrition')  # 替换 'feature_column' 为你要查看的特征列的名称

# 设置图形标签和标题
plt.xlabel("特征值", fontsize=14)
plt.ylabel("出现次数", fontsize=14)
plt.title("特征值出现次数分布", fontsize=16)

# 显示图形
plt.xticks(rotation=45)  # 可选：旋转 x 轴标签，以避免重叠
plt.tight_layout()
plt.show()
# %%
train_df.corr()
# %%
train_df.isna().sum()
# %%
train_df.info()
# %%
train_df.shape
# %%
X = train_df.drop('Attrition',axis=1)
y = train_df['Attrition']
# %%
X
# %%
y
# %%
X_trian,X_val,y_train,y_val = train_test_split(X,y,train_size=0.2,random_state=42)

# %%
X_trian.shape,y_train.shape
# %%
logistic_regression = LogisticRegression(
    penalty = 'l2',
    C = 1.0,
    max_iter=100,
    solver='lbfgs',
    random_state=42
)
# %%
logistic_regression.fit(X_trian,y_train)
# %%
from sklearn.metrics import roc_auc_score
train_score = roc_auc_score(y_train, logistic_regression.predict(X_trian), average='micro')
train_score
# %%
catboost_classifier = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',  
    random_seed=42,
    verbose=100  
)
# %%
catboost_classifier.fit(X_trian, y_train)
train_score = roc_auc_score(y_train, catboost_classifier.predict(X_trian), average='micro')
train_score
# %%
xgb_classifier = xgb.XGBClassifier(
    learning_rate=0.1,
    n_estimators=100,
    max_depth=3,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27
)
# %%
xgb_classifier.fit(X_trian, y_train)
train_score = roc_auc_score(y_train, xgb_classifier.predict(X_trian), average='micro')
train_score
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
    X_trian,y_train,
    validation_data=(X_val,y_val),
    batch_size=16,
    epochs=20,
    callbacks=[early_stopping]

)
history = pd.DataFrame(training.history)
history.loc[:, ['loss', 'val_loss']].plot()
# %%
from sklearn.preprocessing import OrdinalEncoder
object_cols = [col for col in X_trian.columns if X_trian[col].dtype == 'object']
object_cols
# %%
