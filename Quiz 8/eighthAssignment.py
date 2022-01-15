import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = pd.read_csv("./quiz_data.csv")
target = data.loc[:,"Y"]
data = data.drop(["Y"], axis=1)

plt.scatter(data[(target == 1)].X1, data[(target == 1)].X2, c="red", marker="o")
plt.scatter(data[(target == 0)].X1, data[(target == 0)].X2, c="blue", marker="o")
plt.show()

# scaler = StandardScaler()
# scaler = scaler.fit(data)
# data = pd.DataFrame(scaler.transform(data))
# target = pd.DataFrame(scaler.transform(target))

auc = []

clustering_single = AgglomerativeClustering(n_clusters=2, linkage="single").fit(data)
auc.append(accuracy_score(target, clustering_single.labels_))
plt.scatter(data.X1, data.X2, c=clustering_single.labels_, cmap="spring")
plt.title("Hierarchical clustering with single linage")
plt.show()

clustering_complete = AgglomerativeClustering(n_clusters=2, linkage="complete", distance_threshold=None, compute_distances=True).fit(data)
auc.append(accuracy_score(target, clustering_complete.labels_))
plt.scatter(data.X1, data.X2, c=clustering_complete.labels_, cmap="spring")
plt.title("Hierarchical clustering with complete linage")
plt.show()



EPS = [0.75, 1, 1.25, 1.5]
for eps in EPS:
    clustering_dbscan = DBSCAN(eps=eps, min_samples=5).fit(data)
    auc.append(accuracy_score(target, clustering_dbscan.labels_))
    plt.scatter(data.X1, data.X2, c=clustering_dbscan.labels_, cmap="spring")
    plt.title(f"DBSCAN(eps = {eps}, minPts=5)")
    plt.show()

kmeans = KMeans(n_clusters=2).fit(data)
auc.append(accuracy_score(target, kmeans.labels_))
plt.scatter(data.X1, data.X2, c=kmeans.labels_, cmap="spring")
plt.title("kmeans")
plt.show()

print(auc)






