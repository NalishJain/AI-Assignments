# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo 
import bnlearn as bn
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# %%
# fetch dataset 
dataset = fetch_ucirepo(id=109) 
  
# data (as pandas dataframes) 
Xt = dataset.data.features 
yt = dataset.data.targets 



# %%
# Import necessary libraries
df = pd.concat([Xt, yt], axis=1)
df = df.reset_index(drop=True)

divider = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')
X_transformed = divider.fit_transform(df.drop('class', axis = 1))



# %%
df.columns

# %%
df = pd.DataFrame(X_transformed, columns=['Alcohol', 'Malicacid', 'Ash', 'Alcalinity_of_ash', 'Magnesium',
       'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols',
       'Proanthocyanins', 'Color_intensity', 'Hue',
       '0D280_0D315_of_diluted_wines', 'Proline'])

df = pd.concat([df, yt], axis=1)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)


# %%
df.head()

# %%
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

df_train.info()

# %%
network_A = bn.structure_learning.fit(df_train, methodtype='hc', scoretype='k2')
bn.plot(network_A, node_color='#8A0707')

# %%
network_B = bn.independence_test(network_A, df_train, alpha=0.0001, prune=True)
bn.plot(network_B)

# %%
network_A_imp = bn.structure_learning.fit(df_train, methodtype='hc', scoretype='bdeu')
bn.plot(network_A_imp)

# %%

# Feature selection
X = df_train.drop('class', axis=1)
y = df_train['class']

X_test = df_test.drop('class', axis=1)
y_test = df_test['class']

k_best = SelectKBest(f_classif, k=6).fit(X, y)
X_new = k_best.transform(X)

selected_indices = k_best.get_support(indices=True)
selected_features = X.columns[selected_indices]
print("Selected Features:", selected_features)

X_test_new = k_best.transform(X_test)
# Constructing network on selected features
df_new = pd.DataFrame(X_new, columns=selected_features)
df_new['class'] = y

df_test_new = pd.DataFrame(X_test_new, columns=selected_features)
df_test_new['class'] = y_test

network_C = bn.structure_learning.fit(df_new, methodtype='hc', scoretype='bdeu')

# Visualize the network
bn.plot(network_C)



# %%
print(len(network_A['model_edges']),len(network_A_imp['model_edges']),len(network_B['model_edges']),len(network_C['model_edges']))


# %%
# network_A_para = bn.parameter_learning.fit(network_A, df_train, verbose=3)
network_A_imp_para = bn.parameter_learning.fit(network_A_imp, df_train, verbose=3)
network_B_para = bn.parameter_learning.fit(network_B, df_train, verbose=3)
network_C_para = bn.parameter_learning.fit(network_C, df_new, verbose=3)




# %%
CPDs = bn.print_CPD(network_C_para)


# %%
CPDs['class']


# %%

import numpy as np
X, Y = np.meshgrid([0,1,2,3],[1,2,3])

P = np.zeros((3,4))
count = 0
for i in range(3):
    for j in range(4):
        P[i,j] = CPDs['class']['p'][count]
        count+=1



fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, P, cmap = 'Greys')
ax.set_xticks([0,1,2,3])
ax.set_yticks([1,2,3])
ax.set_xlabel('F1 = Flavanoids')
ax.set_ylabel('Class')
ax.set_zlabel('P(Class|F1,F2)')
ax.set_title('3D Scatter Plot')


# %%
df_test.head()

# %%
q1 = bn.inference.fit(network_A_imp_para, variables=['class'], evidence={'Malicacid' : 1, 'Flavanoids' : 2})
q2 = bn.inference.fit(network_A_imp_para, variables=['class'], evidence={'Alcohol' : 2, 'Flavanoids' : 0})
q3 = bn.inference.fit(network_A_imp_para, variables=['class'], evidence={'Malicacid' : 0, 'Flavanoids' : 2, 'Total_phenols' : 2})
q4 = bn.inference.fit(network_A_imp_para, variables=['class'], evidence={'Malicacid' : 0, 'Flavanoids' : 1, 'Magnesium':0})

# %%
prediction_C = bn.predict(network_C_para, df_test_new, variables=['class'])
prediction_A_imp = bn.predict(network_A_imp_para, df_test, variables=['class'])
prediction_B = bn.predict(network_B_para, df_test, variables=['class'])



# %%
print(accuracy_score(y_test,prediction_A_imp['class'].values), accuracy_score(y_test,prediction_B['class'].values), accuracy_score(y_test,prediction_C['class'].values))


# %% [markdown]
# 0.9577464788732394 1.0 0.9929577464788732
# 


