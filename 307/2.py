#%%
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter('ignore')

train = pd.read_csv('train.csv', index_col='id')
test = pd.read_csv('test.csv', index_col='id')
sample = pd.read_csv('sample_submission.csv')

features = test.columns.to_list()
target = 'booking_status'
# %%
from lightgbm import LGBMClassifier

model = LGBMClassifier(n_jobs=-1, verbose=-1, random_state=42)
model.fit(train[features], train[target])
predictions = model.predict(test[features])
sample[target] = predictions
sample.to_csv(f"sample_submission.csv", index=False)
sample
# %%
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

X_train, X_test, y_train, y_test = train_test_split(train[features], train[target], test_size=0.2, random_state=42)
y_pred = model.predict(X_test)
display(pd.crosstab(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f"Precision (mean): {model.score(X_test, y_test)*100:.3}%")
print(f"ROC-AUC: {roc_auc_score(y_test, model.predict(X_test))*100:.3}%")
# %%
