#预测cost
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_squared_error,mean_squared_log_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
# %%
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.head()
# %%
test.head
# %%
train.describe()
# %%
print(f"train shape before : {train.shape}\n")
train_null_counts = train.isnull().sum()
print(f"train null counts before : \n{train_null_counts}\n")
# %%
# Extract features (X) by excluding the last column
X = train.iloc[:, :-1]

# Extract the target variable (Y) as the last column
Y = train.iloc[:, -1]

# %%
X
# %%
Y
# %%
gbm = GradientBoostingRegressor()
gbm.fit(X,Y)
importances_gbm = gbm.feature_importances_

top_features_gbm = pd.Series(importances_gbm, index=X.columns).nlargest(8).index.tolist()
print("Top 8 features from GradientBoostingRegressor:", top_features_gbm)
# %%
most_important_features = ['total_children', 'num_children_at_home',
                           'avg_cars_at home(approx).1', 'store_sqft',
                           'coffee_bar', 'video_store', 'salad', 
                           'florist']

def preprocess(df):
    df = df.copy()
    df['store_sqft'] = df['store_sqft'].astype('category')
    df['salad'] = (df['salad_bar'] + df['prepared_food']) / 2
    df['log_cost'] = np.log1p(df['cost'])
    return df
# %%
train_data = preprocess(train)
X_train, X_test, y_train, y_test = train_test_split(train_data[most_important_features], train_data["log_cost"], test_size=0.2, random_state=2023)
# %%
xgb_params = {'n_estimators': 280,
              'learning_rate': 0.05,
              'max_depth': 10,
              'subsample': 1.0,
              'colsample_bytree': 1.0,
              'tree_method': 'hist',
              'enable_categorical': True,
              'verbosity': 1,
              'min_child_weight': 3,
              'base_score': 4.6,
              'random_state': 2023}

model = xgb.XGBRegressor(**xgb_params)
model.fit(X_train, y_train)
# %%
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Root Mean Squared Error for XGBoost: {mse:.2f}")
# %%
RandomForestRegressor
rf_params = {
     'n_estimators': 100,
     'max_depth': 10,
     'random_state': 2023
 }

rf_model = RandomForestRegressor(**rf_params)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_rf, squared=False)
print(f"Mean Squared Error for rf: {mse:.2f}")
# %%
# CatBoostRegressor
catboost_params = {
    'iterations': 100,
    'learning_rate': 0.1,
    'depth': 6,
    'loss_function': 'RMSE',
    'random_seed': 2023
}

catboost_model = CatBoostRegressor(**catboost_params)
catboost_model.fit(X_train, y_train)

# %%
