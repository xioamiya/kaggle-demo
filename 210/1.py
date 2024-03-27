#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, Lasso, Ridge, LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
# %%
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
games = pd.read_csv('games.csv')
turns = pd.read_csv('turns.csv')
print(games.shape)
print(turns.shape)
print(train.shape)
print(test.shape)
# %%
train.isnull().sum().sum()
# %%
turns.isnull().sum().sum()
# %%
# There also aren't any missing values in test data
# we know rating for different bots and it's neccesary to predict it for people
test["player_type"] = test["nickname"].apply(lambda nick: "bot" if "Bot" in nick else "human")
print(f"Amount of missing values in test data: {test[test['player_type'] == 'bot'].isnull().sum().sum()}")
# %%
test
# %%
nan_columns = turns.columns[turns.isnull().sum()>0]
turns[nan_columns].isnull().sum()
# %%
np.unique(turns[turns['rack'].isna()]["turn_type"], return_counts=True)
# %%
rack_imputer = SimpleImputer(strategy="constant", fill_value="")
turns.loc[:, 'rack'] = rack_imputer.fit_transform(turns['rack'].values.reshape(-1, 1))
# %%
np.unique(turns[turns['location'].isna()]["turn_type"], return_counts=True)
# %%
train["player_type"] = train["nickname"].apply(lambda nick: "bot" if "Bot" in nick else "human")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(data=train, x="rating", bins=40, ax=axes[0], hue="player_type")
axes[0].set_title("Distribution of target value")

sns.boxplot(data=train, x="rating", ax=axes[1], y="player_type")
# %%
print(f"The amount of unique human players names: {train[train['player_type'] == 'human']['nickname'].nunique()}")
# %%
# Let's create a copy of sourse data
games_data = games.copy()
# %%
games_data["created_at"] = pd.to_datetime(games_data["created_at"])
games_data["created_at_month"] = games_data["created_at"].dt.month
games_data["created_at_day"] = games_data["created_at"].dt.day
games_data["created_at_hour"] = games_data["created_at"].dt.hour
games_data["created_at_day_of_week"] = games_data["created_at"].dt.dayofweek
games_data["created_at_is_weekend"] = (games_data["created_at_day_of_week"] > 4).astype(int)
games_data["created_at_timestamp"] = games_data["created_at"].values.astype(np.int64) // 10 ** 9
# %%
periods_of_day = {"M": list(range(5, 11)), "D": list(range(11, 18)), "E": list(range(18, 24)), "N": list(range(0, 5))}  

def select_period_of_day(start_hour, times_of_day=periods_of_day):
    for period, hours in periods_of_day.items():
        if start_hour in hours:
            return period
    return "undefined"
games_data["created_at_period_of_day"] = games_data["created_at_hour"].apply(lambda hour: select_period_of_day(hour))
# %%
games_data = games_data.drop(columns=["created_at"])
# %%
# Now let's take a look at correlations between features
fig, axes = plt.subplots(1)
sns.heatmap(games_data.corr(), ax=axes)
axes.set_title("Correlation between features")
# %%
