import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from load_data import DataLoader
from Preprocessing import DataPreprocessor
from visualize import Visualize

file_path='Dataset/Microsoft_Stock.csv'

loader=DataLoader(file_path)
data=loader.load_file()
print("The loaded data is",data)

preprocessor=DataPreprocessor(data)
preprocessor.clean_and_transform()
cleaned_data=preprocessor.replace_null()
print("The cleaned data is",cleaned_data)

# Visualize data
#viz=Visualize(cleaned_data)
#viz.line_plot()

#Scale data
scaled_data=preprocessor.Normalize(cleaned_data)
print("The scaled data is",scaled_data)

train_date='12/31/2020 16:00:00'
test_date='1/4/2021 16:00:00'
data_split=preprocessor.split_data(scaled_data,train_date,test_date)
print(f'The train shape is {data_split["train"]["input"].shape}')
