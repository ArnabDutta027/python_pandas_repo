import pandas as pd
import numpy as np

date_data = pd.to_datetime("15th January, 2024")
print(date_data)

new_date_data = date_data + pd.to_timedelta(27,unit='D')
print(f" new date - {new_date_data}")

#get all dates for the next 7 days
date_range_fetch = pd.date_range(start = new_date_data,periods=7,freq='D')
print(f"date_range_fetch: \n {date_range_fetch}")