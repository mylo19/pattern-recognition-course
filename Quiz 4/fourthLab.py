import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import  numpy as np

"""""anndata = pd.DataFrame({"X1": [0, 0, 1, 1], "X2": [0, 1, 0, 1], "Y": [1, 1, -1, -1]})
X = anndata.loc[:, ["X1", "X2"]]
y = anndata.loc[:, "Y"]

plt.scatter(anndata[(anndata.Y == -1)].X1, anndata[(anndata.Y == -1)].X2, c="red", marker="+")
plt.scatter(anndata[(anndata.Y == 1)].X1, anndata[(anndata.Y == 1)].X2, c="blue", marker="o")
plt.show()

clf = MLPRegressor(hidden_layer_sizes=(), learning_rate_init=1)
clf = clf.fit(X, y)

print("point [0, 0] ->", clf.predict([[0, 0]]))
print("point [1, 0] ->", clf.predict([[1, 0]]))
print("point [0.5, 0] ->", clf.predict([[0.5, 0]]))

alldata = pd.read_csv("./alldata.txt")
#print(alldata)
xtrain = alldata.loc[0:600, ["X1", "X2"]]
ytrain = alldata.loc[0:600, "y"]
xtest = alldata.loc[600:800, ["X1", "X2"]]
ytest = alldata.loc[600:800, "y"]


plt.scatter(xtrain[(ytrain == 2)].X1, xtrain[(ytrain == 2)].X2, c="red", marker="+")
plt.scatter(xtrain[(ytrain == 1)].X1, xtrain[(ytrain == 1)].X2, c="blue", marker="o")
plt.show() """""

clf = MLPRegressor(hidden_layer_sizes= (500, 20), tol=0.01)
clf = clf.fit(xtrain, ytrain)

"""""
# Training Error
pred = clf.predict(xtrain)
trainingError = [(t - p) for (t, p) in zip(ytrain, pred)]
MAE = np.mean(np.abs(trainingError))
print(MAE)

plt.hist(trainingError, range=(-1,1), rwidth=0.5)
plt.show() """""

# Testing Error
pred = clf.predict(xtest)
testingError = [(t - p) for (t, p) in zip(ytest, pred)]
MAE = np.mean(np.abs(testingError))
print(MAE)

plt.hist(testingError, range=(-1,1), rwidth=0.5)
plt.show()