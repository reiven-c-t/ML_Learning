import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

df = pd.DataFrame([
    ["green", "M", 10.1, "class1"],
    ["red", "L", 13.5, "class2"],
    ["blue", "XL", 15.3, "class1"],
])

df.columns = ["color", "size", "price", "classlabel"]

print(df)

# mapping

size_mapping = {
    "XL": 3,
    "L": 2,
    "M":1
}

df["size"] = df["size"].map(size_mapping)

print(df)

class_mapping = {label: idx for idx, label in enumerate(np.unique(df["classlabel"]))}
df["classlabel"] = df["classlabel"].map(class_mapping)
print(df)

inv_class_mapping = {v: k for k,v in class_mapping.items()}
df["classlabel"] = df["classlabel"].map(inv_class_mapping)
print(df)

class_le = LabelEncoder()
class_le.fit_transform(df["classlabel"].values)
# class_le.inverse_transform(df["classlabel"].values)

# one hot comp

X = df[["color", "size", "price"]].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:,0])
print("item: ",X)

# One hot 2
ohe = OneHotEncoder(categorical_features=[0])
print(ohe.fit_transform(X).toarray())

print(pd.get_dummies(df[["price", "color", "size"]]))

# train_test_split (pass because already used and understand)


