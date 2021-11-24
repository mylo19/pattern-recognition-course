import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt

data = pd.read_csv("./quiz_data.csv")
model1 = data.loc[:, ['P_M1']]
model2 = data.loc[:, ['P_M2']]
classDerived = data.loc[:, ['Class']]

fpr1, tpr1, thresholds1 = roc_curve(classDerived, model1)
print(tpr1)
print(thresholds1)

model2Binary = round(model2)
fscore = f1_score(classDerived, model2Binary, pos_label=1)
print(fscore)

fpr2, tpr2, thresholds2 = roc_curve(classDerived, model2)
print("AUC: ", auc(fpr2, tpr2))

plt.title('Receiver Operating Characteristic')
plt.plot(fpr1, tpr1, 'b', label = 'AUC = %0.2f' % auc(fpr1, tpr1))
plt.plot(fpr2, tpr2, 'b', label = 'AUC = %0.2f' % auc(fpr2, tpr2), c='green')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()