import pandas as pd
import numpy as np

df = pd.read_csv("salary_data.csv")
df["Salary"] = df["Salary"].str.replace("$", "").str.replace(" ", "").str.replace(",","")
df["Salary"] = df["Salary"].astype(float).apply(np.ceil).astype(int)
df = df.groupby(["Name"],as_index = False).sum()
#df2 = df["Name","Title"].groupby("Name", as_index=False)
#print(df2)
df.to_csv('salary_data_cleaned.csv',index=False)

