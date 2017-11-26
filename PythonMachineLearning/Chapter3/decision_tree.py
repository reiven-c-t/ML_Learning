from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  # fix "from sklearn.cross_validation" because ver 1.8 warned
from sklearn.tree import DecisionTreeClassifier
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

tree = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=0)
tree.fit(X_train_std, y_train)

y_pred = tree.predict(X_test_std)
import matplotlib.pyplot as plt
from PythonMachineLearning.Chapter3.plot_decision import plot_desicion_regions
plot_desicion_regions(X=X_test_std, y=y_test, classifier=tree)
plt.show()
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

#rom sklearn.tree import export_graphviz
#export_graphviz(tree, out_file="tree.dot", feature_names=["petal length", "petal width"])