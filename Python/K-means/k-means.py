import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]

if getattr(sys, "frozen", False):
    base_dir = os.path.dirname(sys.executable)
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))
iris_path = os.path.join(base_dir, "iris.csv")

iris = pd.read_csv(iris_path, header=None)

iris = iris.to_numpy(copy=True)

temp = iris[:, :4]
X = np.array(temp, dtype="float32")
Y = iris[:, 4]

label_map = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
Y = pd.Series(Y).map(label_map).astype(int).to_numpy()


class KMeans:
    def __init__(self, k=3, maxIter=100, tol=1e-4, randomState=None):
        self.k = k
        self.maxIter = maxIter
        self.tol = tol
        self.randomState = randomState
        self.centroids = None
        self.labels = None

    def initializeCentroids(self, X):
        np.random.seed(self.randomState)
        index = np.random.choice(len(X), self.k, replace=False)  # 随机抽取k个中心点
        return X[index]

    def computeDistances(self, X):
        return np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)

    def fit(self, X):
        self.centroids = self.initializeCentroids(X)

        for _ in range(self.maxIter):
            distances = self.computeDistances(X)
            self.labels = np.argmin(distances, axis=1)
            oldCentroids = self.centroids.copy()

            for i in range(self.k):
                clusterPoints = X[self.labels == i]
                if len(clusterPoints) > 0:
                    self.centroids[i] = clusterPoints.mean(axis=0)

            if np.linalg.norm(self.centroids - oldCentroids) < self.tol:
                break

    def predict(self, X):
        if self.centroids is None:
            raise ValueError("Model Not Fitted.")
        distances = self.computeDistances(X)
        return np.argmin(distances, axis=1)


kmeans = KMeans(k=3, randomState=1)
kmeans.fit(X)
labels = kmeans.labels

fig = plt.figure(figsize=(12, 8))

ax = fig.add_subplot(111, projection="3d")
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y, cmap="viridis", s=20)
ax.scatter(
    kmeans.centroids[:, 0],
    kmeans.centroids[:, 1],
    kmeans.centroids[:, 2],
    c="red",
    marker="X",
    s=200,
    label="中心",
)

test_cases = np.array(
    [
        [5.3, 3.5, 1.4, 0.2],
        [7.2, 3.2, 4.7, 1.4],
        [6.3, 3.3, 6.2, 2.5],
        [5.0, 3.3, 1.4, 0.2],
        [6.7, 3.1, 4.1, 1.4],
        [5.9, 3.0, 5.1, 1.8],
        [4.5, 3.0, 1.4, 0.2],
        [6.4, 3.0, 4.5, 1.5],
        [5.8, 2.7, 5.4, 1.9],
        [5.7, 2.4, 4.1, 1.3],
    ]
)


test_labels = kmeans.predict(test_cases)

ax.scatter(
    test_cases[:, 0],
    test_cases[:, 1],
    test_cases[:, 2],
    c=test_labels,
    cmap="viridis",
    marker="^",
    s=100,
    edgecolor="k",
    label="测试用例",
)

plt.title("K-means 聚类结果")
plt.legend()
plt.show()
