import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)
df_wine.columns = [
    "Class label",
    "Alcohol",
    "Malic acid",
    "Ash",
    "Alcalinity of ash",
    "Magnesium",
    "Total phenols",
    "Flavanoids",
    "Nonflavanoid phenols",
    "Proanthocyanins",
    "Color intensity",
    "Hue",
    "OD280/OD315 of diluted wines",
    "Proline"
]

print("Class labels", np.unique(df_wine["Class label"]))

# train_test_split

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].value
y = y.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
y_train_std = stdsc.fit_transform(y_train)
X_test_std = stdsc.fit_transform(X_test)

cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print(eigen_vals)

tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

import matplotlib.pyplot as plt

plt.bar(range(1, 14), var_exp, alpha=0.5, align="center", label="individual explained variance")
plt.step(range(1, 14), cum_var_exp, where="mid", label="cumulative explained variance")
# plt.show()

# feature transformation
eigen_pairs =[(np.abs(eigen_vals[i]),eigen_vecs[:,i])
             for i in range(len(eigen_vals))]
eigen_pairs.sort(reverse=True)

w= np.hstack((eigen_pairs[0][1][:, np.newaxis],
              eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W:\n',w)
X_train_pca = X_train_std.dot(w)
