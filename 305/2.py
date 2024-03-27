#%%
import pandas as pd
import numpy as np
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torchmetrics import CohenKappa
from torch.utils.data import Dataset, DataLoader

# %%
df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df.head()
# %%
print('Length of train set:', len(df))
print('Length of test set:', len(df_test))
print('Number of missing values in train set:', df.isnull().any().sum())
print('Number of missing values in test set:', df_test.isnull().any().sum())
print(df.dtypes)
# %%
df.quality.value_counts().sort_index().plot.bar()
# %%
def get_score(y1, y2):
    score = cohen_kappa_score(y1, y2, weights='quadratic')
    return score
# %%
X = df.copy()
y = X.pop('quality')
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
# %%
class MyData(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.int32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
# %%
traindata = MyData(X_train, y_train)
traindata[0]
# %%
trainloader = DataLoader(traindata, batch_size=4, shuffle=True)
# %%
# View a sample
trainiter = iter(trainloader)
print(next(trainiter))
# %%
y_train.nunique()
# %%
input_dim = X_train.shape[1]
hidden_dim = 15
output_dim = y_train.nunique()
class MyNN(nn.Module):
    def __init__(self):
        super(MyNN, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = self.linear2(x)
        return x
    
clf = MyNN()  
print(clf.parameters)
# %%
criterion = CohenKappa(task='multiclass', num_classes=y_train.nunique())
optimizer = torch.optim.SGD(clf.parameters(), lr=0.1)
# %%
criterion(torch.tensor([1,2]), torch.tensor([1,0]))
# %%
epochs = 15
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        inputs, labels = data
        # set optimizer to zero grad to remove previous epoch gradients
        optimizer.zero_grad()
        # forward propagation
        outputs = clf(inputs)
        print('outputs:', outputs)
        print('labels:', labels)
        loss = criterion(outputs, labels)
        # backward propagation
        loss.backward()
        # optimize
        optimizer.step()
        running_loss += loss.item()
        # display statistics
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}')
# %%
