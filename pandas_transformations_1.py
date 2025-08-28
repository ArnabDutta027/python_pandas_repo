import pandas as pd
import numpy as np
from ast import literal_eval
import ast
import json

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


#rename columns
columns_renamed_dict={}
for column_names in product_df.columns:
    columns_renamed_dict.update({column_names: str(column_names).upper()})

product_df = product_df.rename(columns=columns_renamed_dict)
print(product_df.head())


#convert dataframe to numpy array
product_numpy_array=product_df.to_numpy()
print(product_numpy_array)
print(type(product_numpy_array))

for rows in product_numpy_array:
    row_list = rows.reshape(-1).tolist()
    type_rows = type(rows)
    print(f"PRODUCT - {row_list[0]} ,SALES - {row_list[1]}, SALES_SQUARE - {row_list[2]}, SALES_MULTIPLIED - {row_list[3]}, ADD_SALES_SQUARE_AND_SALES_MULTIPLIED - {row_list[4]} , TYPE OF ROWS - {type_rows}")

#eval
tot="4+3"
print(eval(tot))

sample_json_string="""{"name": "John", "age": 30, "city": "New York"}"""
print(eval(sample_json_string))

sample_json_string_1 = """{
"order_num" : "O2012019231a",
"order_date" : "2012-06-27",
"order_id" : 21934,
"order_item" : [
{
"product_id" : 20933,
"quantity" : 3,
"price" : 36000,
"product_name" : "Thingamagic 2000",
"unit_price" : 12000
},
{
"product_id" : 10366,
"quantity" : 1,
"price" : 100,
"product_name" : "Super Duper Blooper",
"unit_price" : 100
}
]
}"""
print(eval(sample_json_string_1))
#ast.literal_eval


sample_json_string_2 = """{"coord":{"lon":-0.1257,"lat":51.5085},
"weather":[{"id":801,"main":"Clouds","description":"few clouds","icon":"02d"}],
"base":"stations",
"main":{"temp":292.86,"feels_like":287.94,"temp_min":292.04,"temp_max":293.71,"pressure":1027,"humidity":45},
"visibility":10000,"wind":{"speed":6.17,"deg":260},"clouds":{"all":23},"dt":1617027097,"sys":{"type":1,"id":1414,"country":"GB","sunrise":1616996538,"sunset":1617042474},"timezone":3600,"id":2643743,"name":"London","cod":200}"""
parsed_Value= ast.literal_eval(sample_json_string_2)
print(parsed_Value.keys())
#pd.json_normalize

json.loads(sample_json_string_2)
df = pd.json_normalize(json.loads(sample_json_string_2),max_level=1)
print(df.head())
print(df.columns)


df_1 = pd.json_normalize(ast.literal_eval(sample_json_string_2),max_level=1)
print(df_1.head())
print(df_1.columns)



df_2 = pd.json_normalize(eval(sample_json_string_2),max_level=1)
print(df_2.head())
print(df_2.columns)