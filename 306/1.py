#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import plotly.express as px
from pandas.plotting import scatter_matrix
matplotlib.rcParams['figure.figsize'] = (12,8)
from scipy import stats

from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV,HalvingGridSearchCV,HalvingRandomSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

import time
# %%
train = pd.read_csv('train.csv',index_col='id')
test = pd.read_csv('test.csv',index_col='id')
submission = pd.read_csv('sample_submission.csv')

trained = train.copy()
tested = test.copy()

# %%
train.head()
# %%
train.shape
# %%
test.shape
# %%
print(train.isnull().sum().sum(),test.isnull().sum().sum())
# %%
train.head()
# %%
def new_features(df):
    data = df.copy()
    data['age'] = df['made'].apply(lambda x:2023-x) # Flat age
    data['meter'] = df['squareMeters'].apply(lambda x:x**0.5) # Square root of area of flat
    data['AreaPerRoom'] = df['squareMeters'] / df['numberOfRooms'] # Average area for a room
#     data['AverageStay'] = data['age']/df['numPrevOwners']
    data.drop(['made','cityCode'],axis=1,inplace=True)
    return data
# %%
train_data = new_features(trained)
test_data = new_features(tested)
# %%
train
# %%
train_data
# %%
