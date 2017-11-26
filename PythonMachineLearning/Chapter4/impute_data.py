import pandas as pd
from io import StringIO
from sklearn.preprocessing import Imputer

csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
0.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))
print(df)

imr = Imputer(missing_values="Nan", strategy="mean", axis=0)
imr = imr.fit(df)
imputed_data = imr.transform(df.values)
print(imputed_data)