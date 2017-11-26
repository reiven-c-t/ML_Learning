from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  # fix "from sklearn.cross_validation" because ver 1.8 warned
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

print(type(iris.data))
print(iris.data[0:10])

X = iris.data[:, [2, 3]]  # maybe this is septal and petal?
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

ppn = Perceptron(max_iter=40, eta0=0.1, random_state=0)  # fix n_iter to max_iter
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))