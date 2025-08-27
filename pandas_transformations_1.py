import pandas as pd
import numpy as np

def multiply_values(rows):
    return rows["calories"] * rows["duration"]

data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}

df = pd.DataFrame(data)
print(df.head())
df['multiplied_values'] = df[["calories","duration"]].apply(multiply_values,axis=1)
print(df.head())