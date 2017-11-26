from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  # fix "from sklearn.cross_validation" because ver 1.8 warned
from sklearn.metrics import accuracy_score
import numpy as np


from PythonMachineLearning.Chapter3.plot_decision import plot_desicion_regions
iris = datasets.load_iris()

X = iris.data[:, [2, 3]]  # maybe this is septal and petal?
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

svm = SVC(kernel="rbf", C=1.0, random_state=0, gamma=0.2)
svm.fit(X_train_std, y_train)



y_pred = svm.predict(X_test_std)

import matplotlib.pyplot as plt
plot_desicion_regions(X=X_test_std, y=y_test, classifier=svm)
plt.show()
print('Accuracy: %.9f' % accuracy_score(y_test, y_pred))