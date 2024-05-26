# %%
from ucimlrepo import fetch_ucirepo 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
import pandas as pd
import numpy as np


# %%
  
# fetch dataset 
adult = fetch_ucirepo(id=2) 
  
# data (as pandas dataframes) 
X = adult.data.features 
y = adult.data.targets 
  

  

# %%
df = pd.concat([X, y], axis=1)
df = df.reset_index(drop=True)


df = pd.get_dummies(df, columns= ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'])
df['income']
X_all = df.drop('income', axis =1).values
y_all = df['income'].values
scaler = StandardScaler()
X_all = scaler.fit_transform(X_all)

for i in range(y_all.shape[0]):
    if y_all[i] == '<=50K' or y_all[i] == '<=50K.':
        y_all[i] = 0
    else:
        y_all[i] = 1
y_all = np.array(y_all, dtype='float')


# %%
model = LogisticRegression()
model.fit(X_all, y_all)
y_all_pred = model.predict(X_all)
full_acc = accuracy_score(y_all, y_all_pred)

# %%
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
train_acc = accuracy_score(y_train_pred, y_train)
test_acc = accuracy_score(y_test_pred, y_test)

# print(train_acc, test_acc)

# %%
accs = []
for i in range(0,20):
    X_train, X_temp, y_train, y_temp = train_test_split(X_all, y_all, test_size=0.3, random_state=i)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=i)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    train_acc = accuracy_score(y_train_pred, y_train)
    val_acc = accuracy_score(y_val_pred, y_val)
    test_acc = accuracy_score(y_test_pred, y_test)
    accs.append(test_acc)
accs = np.array(accs)
test_mean = np.mean(accs)
test_std = np.std(accs)
# print(test_mean, test_std)

# %%
print(f'Full dataset accuracy: full: {full_acc}, train: {train_acc}, test: {test_acc}')
print(f'70-15-15 Cross validation boxplot: mean={test_mean}, std={test_std}')


