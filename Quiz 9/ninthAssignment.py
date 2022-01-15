import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

data = pd.read_csv("quiz_data.csv")
target = data.loc[:, "Y"]
data = data.drop(["Y"], axis=1)

plt.scatter(data.X1, data.X2, marker="o")
plt.title('Data')
plt.show()

plt.scatter(data[(target == 1)].X1, data[(target == 1)].X2, c='red', marker="o")
plt.scatter(data[(target == 2)].X1, data[(target == 2)].X2, c='blue', marker="o")
plt.scatter(data[(target == 3)].X1, data[(target == 3)].X2, c='green', marker="o")
plt.title('True Data')
plt.show()

kmeans = KMeans(n_clusters=3).fit(data)
plt.scatter(data[(kmeans.labels_ == 0)].X1, data[(kmeans.labels_ == 0)].X2, c="red", marker="o")
plt.scatter(data[(kmeans.labels_ == 1)].X1, data[(kmeans.labels_ == 1)].X2, c="blue", marker="o")
plt.scatter(data[(kmeans.labels_ == 2)].X1, data[(kmeans.labels_ == 2)].X2, c="green", marker="o")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker="+", s=169, color="black")
plt.title('Kmeans')
plt.show()

gm = GaussianMixture(n_components=3, tol=0.0001).fit(data)
clusters = gm.predict(data)
plt.scatter(data[(clusters == 0)].X1, data[(clusters == 0)].X2, c='red')
plt.scatter(data[(clusters == 1)].X1, data[(clusters == 1)].X2, c='blue')
plt.scatter(data[(clusters == 2)].X1, data[(clusters == 2)].X2, c='green')
plt.scatter(gm.means_[:, 0], gm.means_[:, 1], marker="+", s=169, color="black")
plt.title('Gaussian Mixture')
plt.show()
# print(gm.covariances_)
print(gm.means_)
