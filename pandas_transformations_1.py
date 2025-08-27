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

## using lambda function

df["multiplied_values_lambda"] = df[["calories","duration"]].apply(lambda rows: rows["calories"] * rows["duration"], axis=1)
print(df.head())
 

# Create a Data Frame
lst=[['C',45],['A',60],['A',26],['C',57],['C',81]]

product_df=pd.DataFrame(lst,columns=['Product','Sales'])
print(product_df.head())

#filter on product having sale =45
print(product_df[product_df['Sales'] == 45])

#filter on product having sale >=60 using query
print(product_df.query('Sales >= 60'))

#get just true or false
print(product_df['Sales'].eq(45))

# using map function

product_df["sales_square"] = product_df["Sales"].map(lambda x: x**2)
print(product_df.head())

def multiply_values_sales(sales):
    return sales*100

product_df['sales_multiplied'] = product_df["Sales"].map(multiply_values_sales)
print(product_df.head())

def add_values(x,y):
    return x+y

product_df['add_sales_square_and_sales_multiplied'] = product_df[["sales_square","sales_multiplied"]].apply(lambda rows: add_values(rows["sales_square"],rows["sales_multiplied"]), axis=1)
print(product_df.head())
