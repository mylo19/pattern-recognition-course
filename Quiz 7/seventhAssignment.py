import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import math
from sklearn.metrics import silhouette_score

data = pd.read_csv("./quiz_data.csv")
data = data.loc[:, ["X1","X2"]]
x1 = [-4, 10]
x2 = [0, 0]
x3 = [4, 10]

initial = np.array((x1,x2,x3))

kmeans = KMeans(n_clusters=3, init=initial).fit(data)
print(kmeans.inertia_)

seperation = 0
distance = lambda x1, x2: math.sqrt(((x1.X1 - x2.X1) ** 2) + ((x1.X2 - x2.X2) ** 2))
m = data.mean()
for i in list(set(kmeans.labels_)):
    mi = data.loc[kmeans.labels_ == i, :].mean()
    Ci = len(data.loc[kmeans.labels_ == i, :].index)
    seperation += Ci * (distance(m, mi) ** 2)
print(seperation)

print(silhouette_score(data, kmeans.labels_))

# change in the inital centers

x1 = [-2, 0]
x2 = [2, 0]
x3 = [0, 10]

initial = np.array((x1,x2,x3))

kmeans = KMeans(n_clusters=3, init=initial).fit(data)
print(kmeans.inertia_)

seperation = 0
distance = lambda x1, x2: math.sqrt(((x1.X1 - x2.X1) ** 2) + ((x1.X2 - x2.X2) ** 2))
m = data.mean()
for i in list(set(kmeans.labels_)):
    mi = data.loc[kmeans.labels_ == i, :].mean()
    Ci = len(data.loc[kmeans.labels_ == i, :].index)
    seperation += Ci * (distance(m, mi) ** 2)
print(seperation)

print(silhouette_score(data, kmeans.labels_))