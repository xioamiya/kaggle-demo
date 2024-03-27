#%%
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 14
plt.rcParams['figure.figsize'] = (22, 5)
plt.rcParams['figure.dpi'] = 100
# %%
train = pd.read_csv('train.csv')
# %%
fig, ax = plt.subplots()
N, bins, patches = ax.hist(np.array(train.avg_glucose_level), edgecolor='white', color='lightgray',linewidth=5, alpha=0.7)
for i in range(1,2):
    patches[i].set_facecolor('orange')
    plt.title('Avg Glucose Level Histogram', fontsize=18)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlabel('avg_glucose_level')
    plt.ylabel('Count')
    plt.axvline(train.avg_glucose_level.mean(), linestyle='--', lw=2, zorder=1, color='blue')
    plt.annotate(f' mean', (90, 7500), fontsize=14,color='black')
    plt.show()
# %%
sns.scatterplot(y=train['bmi'], x=train['avg_glucose_level'], hue=train['stroke'], alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.axvline(train['avg_glucose_level'].mean(), linestyle='--', lw=2, zorder=1, color='black')
plt.annotate(f'mean', (80, 80), fontsize=14, color='black')

plt.title('avg_glucose_level & bmi relation', fontsize=18)
plt.xlabel('avg_glucose_level')
plt.ylabel('bmi')

plt.show()
# %%
sns.scatterplot(x=train['age'], y=train['avg_glucose_level'], hue=train['stroke'], alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.axhline(train['avg_glucose_level'].mean(), linestyle='--', lw=2, zorder=1, color='red')
plt.annotate(f'avg_glucose_level', (70, 100), fontsize=14, color='black')

plt.title('avg_glucose_level & age relation', fontsize=18)
plt.ylabel('avg_glucose_level')
plt.xlabel('age')
plt.show()
# %%
cols = ['gender', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type', 'smoking_status', 'stroke']
for i in train[cols].columns:
    sns.barplot(data=train,x= i,y='avg_glucose_level', hue='stroke', ci=None)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.axhline(train['avg_glucose_level'].mean(), linestyle='--', lw=2, zorder=1, color='black')
    plt.title(f'Avg Glucose Levels, {i} status wise')
    plt.ylabel('Avg Glucose Level')
    plt.xlabel(i)
    plt.annotate('Avg Glucose Level', (0.2, 80))
    plt.show()
# %%
train.bmi = train.bmi.fillna(round(train.bmi.mean(),2), axis=0)
train_df = train[train.columns[1:-1]]
# %%
cat_df = train_df[['gender', 'ever_married','work_type', 'Residence_type','smoking_status']]
cat_df = cat_df.astype('category')
cat_df = cat_df.apply(lambda x : x.cat.codes)
cat_df.head()
# %%
train_df[cat_df.columns] = cat_df.copy()
train_df.head()
# %%
plt.figure(figsize=(22, 6))
train['stroke'].value_counts().plot(kind='bar')
plt.xlabel('Smoking Status')
plt.ylabel('Count')
plt.title('Target Class Countplot',  fontsize=15)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks(rotation=None)
plt.show()
# %%
X = train_df.values
y = train.stroke.values
# %%
X.shape
# %%
y.shape
# %%
#Resampling
rus = RandomOverSampler(random_state=0)
X_resampled, y_resampled = rus.fit_resample(X,y)
# %%
for i in np.unique(y_resampled):
    class_counts= len(y_resampled[y_resampled==i])
    print(f'Instances of the class {i} after re-sampling : {class_counts}')
# %%
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
# %%
# Initialize parameters
w_input_hidden, b_hidden, w_hidden_output, b_output, w_output_classify, b_classify = initialize_parameters(input_size, hidden_size, output_size)

# Hyperparameters
learning_rate = 0.001
num_epochs = 20

# Lists to store accuracy, loss
accuracy_list = []
loss_list = []
#Dictionary to store weights
Weights = {}
#decay rate for decaying the learning rate over time
decay_rate = 5

for epoch in range(num_epochs):
    # Forward propagation on the entire training set
    a_hidden, a_output, a_classify = forward_propagation(X_train.T, w_input_hidden, b_hidden,
                                                         w_hidden_output, b_output, w_output_classify, b_classify)
    
    # Calculate loss
    loss = compute_loss(y_train.reshape(1, -1), a_classify)
    
    # Calculate accuracy
    accuracy = compute_accuracy(y_train.reshape(1, -1), a_classify)
    
    # Backpropagation
    dw_input_hidden, db_hidden, dw_hidden_output, db_output, dw_output_classify, db_classify = backpropagation(X_train.T, y_train.reshape(1, -1), 
                                                    a_hidden, a_output, a_classify, w_input_hidden, w_hidden_output, w_output_classify)
    
    learning_rate = (1 / (1 + decay_rate)) * learning_rate

    
    # Update parameters
    w_input_hidden, b_hidden, w_hidden_output, b_output, w_output_classify, b_classify = update_parameters(w_input_hidden, b_hidden, w_hidden_output, 
                            b_output, w_output_classify, b_classify, dw_input_hidden, db_hidden, dw_hidden_output, db_output, dw_output_classify, 
                                         db_classify, learning_rate)
    
    # Print accuracy and loss 
    if (epoch + 1) % 4 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs} [==============================] - accuracy : {accuracy:.4f} - loss : {loss:.4f}  - learning_rate : {learning_rate}")
    
    # Save accuracy and loss values
    accuracy_list.append(accuracy)
    loss_list.append(loss)
    
    #Saving the weights at the last epoch
    if epoch == num_epochs-1:
        Weights[f'epoch_{epoch}'] = {
        'w_input_hidden': w_input_hidden,
        'b_hidden': b_hidden,
        'w_hidden_output': w_hidden_output,
        'b_output': b_output,
        'w_output_classify': w_output_classify,
        'b_classify': b_classify
                                    }
# %%
