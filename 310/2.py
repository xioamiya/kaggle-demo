#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.io.formats.style import Styler
from tqdm.auto import tqdm
tqdm.pandas()

train_df = pd.read_csv('train.csv').drop(columns='id')
test_df = pd.read_csv('test.csv').drop(columns='id')
sub_df = pd.read_csv('sample_submission.csv')
#%%
test_df.shape
# %%
### Basic functions for EDA
palette = ['#302c36', '#037d97', '#E4591E', '#C09741',
           '#EC5B6D', '#90A6B1', '#6ca957', '#D8E3E2']

def stylize_simple(df: pd.DataFrame, caption: str): 
    s = df
    s = s.style.set_table_styles([{'selector': 'tr:hover',
      'props': [('background-color', 'yellow')]}]).set_caption(f"{caption}")
    return s

def explore_df(df):
    cols = df.columns
    out = pd.DataFrame({"first_row": df.iloc[1],
                        'type': [df[x].dtype.name for x in cols],
                        "n_unique": [len(df[x].unique()) for x in cols],
                        "n_missing": [sum(df[x].isna()) for x in cols],
                        "min": [min(df[x]) if df[x].dtype.name in ['int64', 'float64'] else np.nan for x in cols],
                        "max": [max(df[x]) if df[x].dtype.name in ['int64', 'float64'] else np.nan for x in cols]})
    return out
# %%
stylize_simple(explore_df(train_df), 'Train')
# %%
stylize_simple(explore_df(test_df), 'Test')
# %%
df = train_df.copy()

df_train = df.sample(frac=0.7, random_state=0)
df_valid = df.drop(df_train.index)
y_train = df_train['Class']
y_valid = df_valid['Class']
X_train = df_train.drop('Class', axis=1)
X_valid = df_valid.drop('Class', axis=1)

max_ = X_train.max(axis=0)
min_ = X_train.min(axis=0)

X_train = (X_train - min_) / (max_ - min_)
X_valid = (X_valid - min_) / (max_ - min_)
X_train.dropna(axis=1, inplace=True) # drop the empty feature in column 2
X_valid.dropna(axis=1, inplace=True)
# %%
test_df2 = test_df.copy()
test_df2 = (test_df2  - min_) / (max_ - min_)
y_valid.shape
# %%
from tensorflow import keras
from keras import layers

# YOUR CODE HERE: define the model given in the diagram
model = keras.Sequential([ 
    layers.Dense(256, activation='relu', input_shape=[X_train.shape[1]]),
    layers.Dropout(0.2),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid'),
])
model.summary()
# %%
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.0001,
    restore_best_weights=True,
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid,y_valid),
    batch_size=256,
    epochs=200,
    callbacks=[early_stopping],
)

history_df = pd.DataFrame(history.history)
history_df.loc[:,['loss','val_loss']].plot(title='Cross-entropy')
history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot(title="Accuracy")
#%%
# 使用模型对验证数据集进行预测
y_pred = model.predict(X_valid)

# 将模型输出的概率值转换为二进制类别（0或1）
y_pred_binary = (y_pred > 0.5).astype(int)

# 导入准确率评估函数
from sklearn.metrics import accuracy_score
#%%
y_pred_binary.shape
#%%
# 计算准确率
accuracy = accuracy_score(y_valid, y_pred_binary)

# 打印准确率
print("模型在验证数据集上的准确率：", accuracy)

# %%
predict_x=model.predict(test_df2) 
classes_x=np.argmax(predict_x,axis=1)
# %%
train_df.Class.mean(), sum(predict_x > 0.5)/len(predict_x) 
# %%
sub_df['Class'] = predict_x
(sub_df.Class > 0.5).value_counts()
sub_df.to_csv('submission_v1.csv', index=False)
# %%
