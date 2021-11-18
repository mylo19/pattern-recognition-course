import pandas as pd
from sklearn.neural_network import MLPRegressor
import numpy as np

X1 = [2, 2, -2, -2, 1, 1, -1, -1]
X2 = [2, -2, -2, 2, 1, -1, -1, 1]
Y = [1, 1, 1, 1, 2, 2, 2, 2]
alldata = pd.DataFrame({"X1": X1, "X2": X2, "Y": Y})
# print(alldata)
xtrain = alldata.loc[:, ["X1", "X2"]]
ytrain = alldata.loc[:, "Y"]
print(ytrain.values)

clf = MLPRegressor(hidden_layer_sizes=(20, 20),
                   max_iter=10000)  # depending on the question, change the hidden layer size
clf = clf.fit(xtrain.values, ytrain.values)

# Training Error
pred = clf.predict(xtrain.values)
trainingError = [(t - p) for (t, p) in zip(ytrain, pred)]
MAE = np.mean(np.abs(trainingError))
print(MAE)

print("point->", clf.predict([[0.8, -0.6]]))