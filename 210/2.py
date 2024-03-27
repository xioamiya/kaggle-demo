#%%
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# %%
train_raw = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
turns = pd.read_csv('turns.csv')
games = pd.read_csv('games.csv')
sample_submission = pd.read_csv('sample_submission.csv')
sample_submission
# %%
# Pull out the bot data and concat it along column axis

bots = ['BetterBot', 'STEEBot', 'HastyBot']
player_df  = train_raw[~train_raw['nickname'].isin(bots)]
bots_df = train_raw[train_raw['nickname'].isin(bots)]
train_raw = pd.merge(player_df, bots_df, on='game_id', suffixes=['_player', '_bot'])
# %%
test_player_df  = test[~test['nickname'].isin(bots)]
test_bots_df = test[test['nickname'].isin(bots)]
test_df = pd.merge(test_player_df, test_bots_df, on='game_id', suffixes=['_player', '_bot'])
# %%
train_raw
# %%
min_max = {}
all_vals = pd.concat([train_raw, test_df])
for col in ['score_player', 'rating_player', 'score_bot', 'rating_bot']:
    min_max[col] = {'min': all_vals[col].min()}
    min_max[col]['max'] = all_vals[col].max()

for df in [train_raw, test_df]:
    for col in ['score_player', 'rating_player', 'score_bot', 'rating_bot']:
        df[col] = (df[col] - min_max[col]['min'])/(min_max[col]['max'] - min_max[col]['min'])
# %%
# Split data frame into validation set

# Split data frame into validation set

train_raw = train_raw.sample(frac=1).reset_index(drop=True)
split = int(train_raw.shape[0]*.1)
train = train_raw.iloc[0:-split]
val = train_raw.iloc[-split:]
# %%
drop_cols = ['game_id', 'nickname_player', 'nickname_bot', 'rating_player']

X_train = train.drop(columns=drop_cols)
y_train = train['rating_player']
X_val = val.drop(columns=drop_cols)
y_val = val['rating_player']
# %%
sns.heatmap(pd.concat([X_train, y_train], axis=1).corr(), annot=True, fmt=".2f")
# %%
def tensor(df):
    return torch.tensor(df.values, dtype=torch.float32)
# %%
X_train = tensor(X_train)
y_train = tensor(y_train)
X_val = tensor(X_val)
y_val = tensor(y_val)
X_train.shape, y_train.shape
# %%
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
# %%
lin_reg.fit(X_train[:, 0].unsqueeze(1), y_train)
# %%
lin_reg.coef_[0], lin_reg.intercept_
# %%
preds = lin_reg.predict(X_val[:, 0].unsqueeze(1))
# %%
from sklearn.metrics import mean_squared_error as mse
mse(preds, y_val)
# %%
x = np.arange(start=0, stop=2, step=1)
y = x * lin_reg.coef_[0] + lin_reg.intercept_

plt.scatter(X_train[:, 0], y_train, c='b')
# plt.scatter(X_train[:, 1], y_train, c='g')
# plt.scatter(X_train[:, 2], y_train, c='r')
plt.plot(x, y, c='k', label='sklearn linreg')
plt.legend()
plt.show
# %%
train_set = torch.utils.data.TensorDataset(X_train, y_train)
val_set = torch.utils.data.TensorDataset(X_val, y_val)
dataset = {'train': train_set, 'val': val_set}
batch_size = 64

data = {
        key: torch.utils.data.DataLoader(
        dataset[key],
        batch_size=batch_size,
        shuffle=True
        ) 
        for key in ['train', 'val']
}
# %%
model = torch.nn.Linear(3, 1)
error = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=1e-2)
model.weight, model.bias.item()
# %%
loss_list = []
# %%
def model_fit(epochs):
    val_size = len(data['val'])
    for epoch in range(epochs):
        model.train()

        for x, y in data['train']:
            optimizer.zero_grad()               
            outputs = model(x)           
            loss = error(outputs, y.unsqueeze(1))    
            loss.backward()                 
            optimizer.step()

        model.eval()
        cumulative_loss = 0
        for x, y in data['val']:
            outputs = model(x)
            eval_loss = error(outputs, y.unsqueeze(1))
            cumulative_loss += eval_loss.item()
        
        loss_list.append(cumulative_loss)
        
        if epoch % 5 == 0:
            print(f'epoch {epoch}. train loss {loss: .3f}, val loss {loss_list[epoch]/val_size: .3f}')
# %%
model_fit(150)

plt.plot(np.arange(len(loss_list)), loss_list, label='val_loss')
# %%
