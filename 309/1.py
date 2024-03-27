#%%
import numpy as np
import pandas as pd
# %%
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
# %%
train_df = pd.read_csv('train.csv')
train_df.head()
# %%
train_df.shape
# %%
train_df.info()
# %%
train_df.drop('id',axis=1,inplace=True)

# %%
train_df.duplicated().sum()
# %%
train_df.isna().sum()
# %%
plt.figure(figsize=(10,5))
sns.histplot(data=train_df, x='Strength',kde=True)
plt.title('Concrete Strength Distribution')
plt.show()
# %%
col_list= train_df.columns.to_list()
col_list
# %%
# Lets check the ditribution of other features
for col in col_list:
    plt.figure(figsize=(13,4))
    plt.subplot(121)
    sns.histplot(data=train_df,x=col,kde=True)
    plt.title("{} Distribution".format(col))
    
    plt.subplot(122)
    stats.probplot(train_df[col],dist='norm', plot=plt)
    plt.title("Q-Q Plot for {}".format(col)) 
    
    plt.show()
# %%
plt.figure(figsize=(20,20))
for i,col in enumerate(col_list[:-1]):    
    plt.subplot(3,3,i+1)
    sns.scatterplot(data=train_df,x=col,y='Strength')
    plt.title("{} vs strength".format(col))
plt.show()
# %%
plt.figure(figsize=(10,7))
sns.heatmap(train_df.corr(),annot=True)
plt.show()
# %%
test_df = pd.read_csv('test.csv')
test_df.head()
# %%
test_data = test_df.drop('id',axis=1)
test_data
# %%
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler,PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,accuracy_score,mean_absolute_error
# %%
X = train_df.drop('Strength',axis=1)
y = train_df['Strength']
# %%
X.head()
# %%
y.head()
# %%
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print(X_train.shape)
print(X_test.shape)
# %%
col_list.remove('Strength')
col_list
# %%
numerical_pipeline = Pipeline([
    ('Imputer',SimpleImputer(strategy='median')),
    ('Power Trf',PowerTransformer()) # By default, zero-mean, unit-variance normalization is applied to the transformed data.
])
# %%
preprocessor = ColumnTransformer([
    ('numerical_features',numerical_pipeline,col_list)
])
# %%
lr = LinearRegression()
# %%
model_pipeline = Pipeline([
    ('Preprocessor',preprocessor),
    ('Linear Reg Model',lr)
])
# %%
model_pipeline.fit(X_train,y_train)
# %%
y_pred_train = model_pipeline.predict(X_train)
r2_score(y_train,y_pred_train)
# %%
scores = cross_val_score(model_pipeline,X_train,y_train,scoring='r2',cv=10,n_jobs=-1)
scores.mean()
# %%
