import pandas as pd
df = pd.read_excel("data.xlsx", dtype=str)   # dtype=str keeps Sanskrit + IDs intact
print(df.shape)
print(df.iloc[:,2])
x=df.iloc[1,2]
print(x)