import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from PythonMachineLearning.Chapter2.load_data import plot_desicion_regions

class Perceptron:
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.weights = np.zeros(1 + X.shape[1])  # 列数(変数数)+定数項(1つ)の数だけ重みを設定&初期化
        self.errors = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):  # 各j行をzipで[...X_j,y_j]みたいな感じにしてxi,とtargetに分ける
                update = self.eta * (target - self.predict(xi))  # ここが $\Delta w_j$ in smallNote.md
                self.weights[1:] += update * xi
                self.weights[0] += update
                errors += int(update != 0.0)

            self.errors.append(errors)

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


# p28
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
print(df.tail())

y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", -1, 1)
X = df.iloc[0:100, [0, 2]].values
plt.scatter(X[:50, 0], X[:50, 1], color="red", marker="o", label="setosa")
plt.scatter(X[50:100, 0], X[50:100, 1], color="blue", marker="x", label="setosa")
plt.xlabel("petal length")
plt.ylabel("sepal length")
plt.legend(loc="upper left")
plt.show()

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors) + 1), ppn.errors, marker="o")
plt.xlabel("Epochs")
plt.ylabel("Number of misclassifications")
plt.show()


plot_desicion_regions(X, y, classifier=ppn)
plt.show()
