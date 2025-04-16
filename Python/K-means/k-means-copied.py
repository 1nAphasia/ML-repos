import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class KMeans:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, random_state=None):
        """
        初始化参数：
        - n_clusters: 簇数量（k）
        - max_iter: 最大迭代次数
        - tol: 中心点变化的容忍度（提前停止条件）
        - random_state: 随机种子
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None

    def _initialize_centroids(self, X):
        """随机初始化中心点（改进：可替换为K-means++）"""
        np.random.seed(self.random_state)
        idx = np.random.choice(len(X), self.n_clusters, replace=False)
        return X[idx]

    def _compute_distances(self, X):
        """计算每个样本到所有中心点的欧氏距离（优化：利用广播机制）"""
        return np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)

    def fit(self, X):
        """训练模型"""
        # 1. 初始化中心点
        self.centroids = self._initialize_centroids(X)
        
        for _ in range(self.max_iter):
            # 2. 计算距离并分配标签
            distances = self._compute_distances(X)
            self.labels_ = np.argmin(distances, axis=1)
            
            # 3. 保存旧中心点用于收敛判断
            old_centroids = self.centroids.copy()
            
            # 4. 更新中心点（按簇均值）
            for i in range(self.n_clusters):
                cluster_points = X[self.labels_ == i]
                if len(cluster_points) > 0:
                    self.centroids[i] = cluster_points.mean(axis=0)
            
            # 5. 检查收敛条件
            if np.linalg.norm(self.centroids - old_centroids) < self.tol:
                break

    def predict(self, X):
        """预测新数据的簇标签"""
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)

# 生成测试数据
X, y_true = make_blobs(
    n_samples=300, 
    centers=3, 
    cluster_std=0.8, 
    random_state=42
)

# 训练模型
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# 可视化结果
plt.figure(figsize=(10, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=20)
plt.scatter(
    kmeans.centroids[:, 0], 
    kmeans.centroids[:, 1], 
    c='red', marker='X', s=200, label='Centroids'
)
plt.title("K-means Clustering Result")
plt.legend()
plt.show()