##################### Cheat Sheet for Clustering #############################################
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("./Quiz 7/quiz_data.csv")
data = data.dropna(how='any', axis=0)

# Y.replace(to_replace="yes", value=1, in_place=True)
# Y.replace(to_replace="no", value=0, in_place=True)

######################## Scaling ###############################################################
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler = scaler.fit(data)
scaled_data = pd.DataFrame(scaler.transform(data), columns=data.columns)

######################### PCA ###################################################################
# PCA (often used with scaled data)
from sklearn.decomposition import PCA

pca = PCA()
pca = pca.fit(data)
pca_transformed = pca.transform(data)
eigenvectors = pca.components_
eigenvalues = pca.explained_variance_
infoLoss = 1 - (eigenvalues[0] + eigenvalues[1]) / sum(eigenvalues)  # find a specific info loss
# of find the infoloss for all posible dimensions kept
# sumInfo = sum(eigenvalues)
# infoLoss = []
# for index in range(len(eigenvalues)):
#     info = sum(eigenvalues[0:index + 1]) / sumInfo
#     infoLoss.append(1 - info)

pca_inverse = pd.DataFrame(pca.inverse_transform(pca_transformed), columns=data.columns)

###################################  KMEANS   ##########################################################

from sklearn.cluster import KMeans

initial = data.iloc[0:3]  # initial points are the first three
kmeans = KMeans(n_clusters=3, init=initial, n_init=1).fit(data)

centers = kmeans.cluster_centers_
labels = kmeans.labels_

# plot the clustered data
plt.scatter(data[(kmeans.labels_ == 0)].X1, data[(kmeans.labels_ == 0)].X2, c="red", marker="o")
plt.scatter(data[(kmeans.labels_ == 1)].X1, data[(kmeans.labels_ == 1)].X2, c="blue", marker="o")
plt.scatter(data[(kmeans.labels_ == 2)].X1, data[(kmeans.labels_ == 2)].X2, c="green", marker="o")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker="+", s=169, color="black")
plt.show()

# Seperation and Inertia (Cohesion)
import math

seperation = 0
distance = lambda x1, x2: math.sqrt(((x1.X1 - x2.X1) ** 2) + ((x1.X2 - x2.X2) ** 2))
m = data.mean()
for i in list(set(kmeans.labels_)):
    mi = data.loc[kmeans.labels_ == i, :].mean()
    Ci = len(data.loc[kmeans.labels_ == i, :].index)
    seperation += Ci * (distance(m, mi) ** 2)

cohesion = kmeans.inertia_

# Cohesion for multiple k to create elbow plot
sse = []
for i in range(1, 11):
    sse.append(KMeans(n_clusters=i, init=data.loc[0:i - 1, :], n_init=1).fit(data).inertia_)
plt.plot(range(1, 11), sse)
plt.scatter(range(1, 11), sse, marker="o")
plt.show()

# silhouette
from sklearn.metrics import silhouette_samples, silhouette_score
from numpy.core.fromnumeric import mean

mean_silhouette_of_all_clusters = silhouette_score(data, kmeans.labels_)
mean_silhouette_of_first_cluster = mean(silhouette_samples(data, kmeans.labels_)[kmeans.labels_ == 0])
mean_silhouette_of_second_cluster = mean(silhouette_samples(data, kmeans.labels_)[kmeans.labels_ == 1])
mean_silhouette_of_third_cluster = mean(silhouette_samples(data, kmeans.labels_)[kmeans.labels_ == 2])

########################### Hierachical Clustering ############################################

from sklearn.cluster import AgglomerativeClustering
import numpy as np
from scipy.cluster.hierarchy import dendrogram

# show dendogram - single linkage
clustering = AgglomerativeClustering(n_clusters=None, linkage="single", distance_threshold=0).fit(data)
linkage_matrix = np.column_stack([clustering.children_, clustering.distances_, np.ones(len(data.index) - 1)]).astype(
    float)
dendrogram(linkage_matrix, labels=labels)
plt.show()

clustering = AgglomerativeClustering(n_clusters=3, linkage="complete").fit(data)
cluster = clustering.labels_
plt.scatter(data[(cluster == 0)].X1, data[(cluster == 0)].X2, c="red", marker="o")
plt.scatter(data[(cluster == 1)].X1, data[(cluster == 1)].X2, c="blue", marker="o")
plt.scatter(data[(cluster == 2)].X1, data[(cluster == 2)].X2, c="green", marker="o")
plt.show()

################################# DBSCAN ####################################################

from sklearn.cluster import DBSCAN

EPS = [0.75, 1, 1.25, 1.5]
for eps in EPS:
    clustering_dbscan = DBSCAN(eps=eps, min_samples=5).fit(data)
    plt.scatter(data.X1, data.X2, c=clustering_dbscan.labels_, cmap="spring")
    plt.title(f"DBSCAN(eps = {eps}, minPts=5)")
    plt.show()

########################## Gaussian Mixture ###################################################

from sklearn.mixture import GaussianMixture

# for 1D data
# data = np.array(x.tolist()).reshape(-1,1)
# gm = GaussianMixture(n_components=2).fit(data)

gm = GaussianMixture(n_components=3, tol=0.1).fit(data)
cluster = gm.predict(data)
centers = gm.means_
