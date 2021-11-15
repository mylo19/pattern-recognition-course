import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
# print(iris)
featureIndex = iris.feature_names.index('petal length (cm)')
data = iris.data
# print(data)
newData = data[:, featureIndex]
meanData = np.mean(newData)
print(meanData)
featureIndex2 = iris.feature_names.index('sepal width (cm)')
newData = data[:, featureIndex2]
maxData = np.max(newData)
print(maxData)
featureIndex3 = iris.feature_names.index('sepal length (cm)')
newData = data[:, featureIndex3]
varData = np.var(newData)
print(varData)

meanC = []
rows, columns = data.shape
for i in range(columns):
    tempData = data[:, i] * 100
    ceilData = round(np.mean(tempData))
    ceilData = ceilData / 100
    meanC.append(ceilData)
print(meanC)
