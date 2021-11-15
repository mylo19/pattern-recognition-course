import pandas as pd
from sklearn.preprocessing import OneHotEncoder

A = [10, 5, 3, 100, -2, 5, -50]
print(A[1:3])

B = []
for index in range(1, 3):
    B.append(A[index])
print(B)

C = [el for el in A if el > 5]
print(C)

D = [index for index, el in enumerate(A) if el > 5]
print(D)

a = [1, 2, 3]
b = [True, True, False]
c = ["a", "b", "c"]
df = pd.DataFrame(list(zip(a, b, c)), columns=['a', 'b', 'c'])
print(df.loc[1, b])

encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)   # Create encoder
encoder.fit(df)
df_one_hot = encoder.transform(df)
print(df_one_hot)