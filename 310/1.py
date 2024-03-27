#%%
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
# %%
train = pd.read_csv('train.csv').drop(columns='id')
test = pd.read_csv('test.csv').drop(columns='id')
train
# %%
test
# %%
train.isna().sum()
# %%
test.isna().sum()
# %%
train.corr
# %%
config = {'SEED': 42,
          'FOLDS': 15,
          'N_ESTIMATORS': 700}

params = {'max_depth': 4,
          'learning_rate': 0.06,
          'colsample_bytree': 0.67,
          'n_jobs': -1,
          'objective': 'binary:logistic',
          'early_stopping_rounds': 150,
          'verbosity': 0,
          'eval_metric': 'logloss'}


X, y = train.drop(columns=['Class']), train.Class
X.shape
y.shape
# %%
from sklearn import model_selection
import xgboost as xgb

cv = model_selection.StratifiedKFold(n_splits=config['FOLDS'],
                                     shuffle=True,
                                     random_state=config['SEED'])

feature_importances_ = pd.DataFrame(index=test.columns)
eval_results_ = {}
models_ = []
oof = []

for fold, (fit_idx, val_idx) in enumerate(cv.split(X, y)):
    if (fold + 1) % 5 == 0 or (fold + 1) == 1:
        print(f'{"#" * 24} Training FOLD {fold + 1} {"#" * 24}')

    X_fit = X.iloc[fit_idx]
    X_val = X.iloc[val_idx]
    y_fit = y.iloc[fit_idx]
    y_val = y.iloc[val_idx]

    fit_set = xgb.DMatrix(X_fit, y_fit)
    val_set = xgb.DMatrix(X_val, y_val)
    watchlist = [(fit_set, 'fit'), (val_set, 'val')]
    