
#import libraries 
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

#importing the dataset

dataset = pd.read_excel('Social.xlsx')
X = dataset.iloc[:,[3,4]].values

#Using elbow method finding the optimal no of clusters

from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot( range(1,11), wcss )
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel("WCSS ")
plt.legend()
plt.show()

#fitting kmeans

kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(X)

#visulazation the clusters

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')


plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')

plt.title("cluster of custmors")
plt.xlabel("Estimate salary")
plt.ylabel("Spending score(1-100)")
plt.legend()
plt.show()
