#%%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from xgboost import XGBClassifier
import catboost
from catboost import CatBoostClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier


#model evaluation
from sklearn.metrics import roc_auc_score

# %%
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
train_df.head()
# %%
train_df.info()
# %%
print(train_df['Class'].describe())

# %%
train_df.corr()
# %%
correlations = train_df.corr()['Class'].drop('Class')
sorted_correlations = correlations.abs().sort_values(ascending=False)
sorted_correlations
# %%
sns.barplot(x=sorted_correlations.index, y=sorted_correlations)
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Absolute Correlation')
plt.show()
# %%
sns.set(style="darkgrid")
# %%
# Plot the distribution of the target variable
sns.countplot(data=train_df, x='Class')
plt.xlabel('Categories')
plt.ylabel('Count')
plt.show()
train_df['Class'].value_counts()
# %%
cols_to_display = train_df.columns[1:]
train_df[cols_to_display].hist(
    figsize=(15, 10), color="blue", edgecolor="black")
plt.suptitle("Numeric columns", fontsize=15)
plt.tight_layout()
plt.show()
# %%
categories = train_df.select_dtypes("object").columns
categories
# %%
# Check and display the number of missing values in each column
train_df.isnull().sum().sort_values(ascending=False)
# %%
test_df.isnull().sum().sort_values(ascending=False)
# %%
# Separate target variable from features
X_train = train_df.drop('Class', axis=1)
y_train = train_df['Class']
X_test = test_df
# %%
sc = StandardScaler()
for col in X_train.columns:
    if col in ["time", "amount"]:
        X_train[col] = sc.fit_transform(X_train[col].values.reshape(-1, 1))
# %%
for col in X_test.columns:
    if col in ["time", "amount"]:
        X_test[col] = sc.transform(X_test[col].values.reshape(-1, 1))
# %%
# Oversampling to balance classes
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)
# %%
X_resampled
# %%
logistic_regression = LogisticRegression(
    penalty = 'l2',
    C = 1.0,
    max_iter=100,
    solver='lbfgs',
    random_state=42
)
# %%
logistic_regression.fit(X_resampled,y_resampled)
# %%
train_score = roc_auc_score(y_train, logistic_regression.predict(X_train), average='micro')

print(f'{logistic_regression.__class__.__name__} micro ROC training score: {train_score:.3f}')
# %%
# Initialize the XGBoost classifier
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
# Training the model on the resampled data
xgb_classifier.fit(X_resampled, y_resampled)
# %%
train_score = roc_auc_score(y_train, xgb_classifier.predict(X_train), average='micro')

print(f'{xgb_classifier.__class__.__name__} micro ROC training score: {train_score:.3f}')

print(f'{xgb_classifier.__class__.__name__} micro ROC training score: {train_score:.3f}')
# %%
# Initialize the CatBoost classifier
catboost_classifier = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',  
    random_seed=42,
    verbose=100  
)
# %%
# Training the model on the resampled data
catboost_classifier.fit(X_resampled, y_resampled)
# %%
train_score = roc_auc_score(y_train, catboost_classifier.predict(X_train), average='micro')

print(f'{catboost_classifier.__class__.__name__} micro ROC training score: {train_score:.3f}')
# %%
params = {
    'learning_rate': 0.1,
    'n_estimators': 200,
    'max_depth': 5,
    'random_state': 42
}
lgb_classifier = lgb.LGBMClassifier(**params)
# %%
# Training the model on the resampled data
lgb_classifier.fit(X_resampled, y_resampled)
# %%
train_score = roc_auc_score(y_train, lgb_classifier.predict(X_train), average='micro')

print(f'{lgb_classifier.__class__.__name__} micro ROC training score: {train_score:.3f}')
# %%
params = {
    'max_depth': 5,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42
}
decision_tree_classifier = DecisionTreeClassifier(**params)
# %%
# Training the model on the resampled data
decision_tree_classifier.fit(X_resampled, y_resampled)
# %%
train_score = roc_auc_score(y_train, decision_tree_classifier.predict(X_train), average='micro')

print(f'{decision_tree_classifier.__class__.__name__} micro ROC training score: {train_score:.3f}')
# %%
