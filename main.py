import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

np.random.seed(42)
X = np.concatenate([np.random.normal(loc=-2, scale=1, size=(100, 2)),
                   np.random.normal(loc=3, scale=1, size=(100, 2)),
                   np.random.normal(loc=7, scale=1, size=(100, 2))])
plt.scatter(X[:,0], X[:,1],s=50,cmap='virdis')
plt.title('Сгнерированные данные')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.show()
kMeans = KMeans(n_clusters=3)
kMeans.fit(X)
labels = kMeans.labels_
centroids = kMeans.cluster_centers_
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', label='Центры кластеров')
plt.title('Результат кластеризации данных')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.show()
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Строим график
plt.plot(range(1, 11), inertia, marker='o')
plt.title("Метод локтя")
plt.xlabel("Число кластеров (k)")
plt.ylabel("Сумма внутрикластерных расстояний")
plt.show()