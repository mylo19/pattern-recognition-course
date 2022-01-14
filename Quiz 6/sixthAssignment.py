import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score

data = pd.read_csv('./quiz_data.csv')
# Define Training and Testing Set
trainingRange = list(range(0, 50)) + list(range(90, 146))
training = data.loc[trainingRange, :]
trainingType = training.loc[:, "Type"]
training = training.drop(["Type"], axis=1)

testingRange = list(range(50, 90))
testing = data.loc[testingRange, :]
testingType = testing.loc[:, "Type"]
testing = testing.drop(["Type"], axis=1)

scaler = StandardScaler()
scaler = scaler.fit(training)
transformed = pd.DataFrame(scaler.transform(training))
testingTransformed = pd.DataFrame(scaler.transform(testing))

pca = PCA()
pca = pca.fit(transformed)
pca_transformed = pca.transform(transformed)
eigenvalues = pca.explained_variance_
# print(eigenvalues)

pc1 = max(eigenvalues)
sumEigen = sum(eigenvalues)

# percentage of information in first principal component
# print(pc1/sumEigen)

sumPC14 = sum(eigenvalues[i] for i in range(4))

# information loss by keeping only the 4 pc
# print(1 - sumPC14/sumEigen)

# find aucurracy and recall score of knn3
clf = KNeighborsClassifier(n_neighbors=3)
clf = clf.fit(training.values, trainingType.values)

pred = clf.predict(testing.values)
auc = accuracy_score(testingType.values, pred)
rec = recall_score(testingType.values, pred, pos_label=2)
# print(rec)

accuracy = []
for i in range(1, 10):
    pca = PCA(n_components=i)
    pca = pca.fit(transformed)
    pca_transformed = pd.DataFrame(pca.transform(transformed))
    pca_testing = pd.DataFrame(pca.transform(testingTransformed))
    # print(pca_transformed)
    clf = KNeighborsClassifier(n_neighbors=3)
    clf = clf.fit(pca_transformed.values, trainingType.values)
    pred = clf.predict(pca_testing.values)
    accuracy.append(accuracy_score(testingType.values, pred))

# find which number of pc kept gives the best accuracy score
answer = accuracy.index(max(accuracy)) + 1
print(answer)