import pandas as pd
from io import StringIO

csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
0.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))
print(df)

print(df.isnull().sum())

"""
基本はdropnaで
na dataを消せる。
なお、C行のnaのみ考慮とか、指定行のみ検査とかなんか色々できる。
"""
#print(df.dropna())