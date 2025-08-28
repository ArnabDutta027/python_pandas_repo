import pandas as pd
import numpy as np

df= pd.DataFrame([10,20,30,40,50],columns=['numbers'])
print(df.head())

print(df.shape)
print(df.columns)
print(df.index)

df["number_multiplied"]= df["numbers"]*100
print(df.head())


student_por_df = pd.read_csv("python_pandas_practice\student-por.csv", sep=";")
print(student_por_df.head())
# print(student_por_df.dtypes)  # Shows columns and their data types
student_por_df = student_por_df.astype({"Medu": float})  # Cast multiple columns to specific types
print(student_por_df.dtypes)

print(f"count of records and columns - {student_por_df.shape}")
print(f"count of - {student_por_df.shape[0]} ")

##selecting subset of columns and rows
student_por_subset_df = student_por_df.loc[:25,["school", "age", "famsize"]]
print(student_por_subset_df.head())
print(f"count of subset - {student_por_subset_df.shape[0]} ")

# group by family size
# print(student_por_df.groupby("famsize")["famsize"].agg(["count"]))

agg_data = student_por_df.groupby("famsize")["famsize"].agg(["count"])
print(f"student agg data: \n  {agg_data}")
index_agg_list_data = list(agg_data.index)
print(f"index_agg_list_data: {index_agg_list_data}")

x = index_agg_list_data
y = agg_data.values
print(f"type of y :{type(y)}")
#flattening a numpy array to list
y  = agg_data.values.reshape(-1).tolist()
print(f"x :{x} \n y: {y}")


# Flattening a nested list
new_array = [[1],[2],[3],[4],[5]]
flattened_list = [elements for sublist_elements in new_array for elements in sublist_elements]
#using numpy arrray to flatten a nested list
print(f"numpy array to list: {np.array(new_array).reshape(-1).tolist()}")

import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.bar(x, y,color='blue', width=0.4)
plt.xlabel("Family Size")
plt.ylabel("Count of Students")
plt.title("Count of Students by Family Size")
plt.show()


# read excel file
excel_df = pd.read_excel("python_pandas_practice\sensor_data.xlsx",sheet_name="20210117_0037",usecols=[2,3,4,5],header=[3],engine='openpyxl')
print(excel_df.head())