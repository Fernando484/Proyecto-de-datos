import pandas as pd

df = pd.read_excel("projects.xlsx")

df.shape                 # (filas, columnas)
df.dtypes                # tipo de cada columna
df.isna().sum()          # ausentes por columna
(df.isna().mean() * 100) # lo mismo, en porcentaje
df.duplicated("projectID").sum()   # duplicados por identificador
df.head()                # primeras filas

# print(df.shape) 
# print(df.dtypes)
# print(df.isna().sum())

## NO SÉ SI ES NECESARIO
# print((df.isna().mean() * 100))
# print(df.duplicated("projectID").sum())
# print(df.head())
