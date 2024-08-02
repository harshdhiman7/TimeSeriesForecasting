import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
      def __init__(self,data):
          self.data=data
          self.data.index=self.data['Date']
          del self.data['Date']
      
      def clean_and_transform(self):
          if self.data.isnull().sum().any():
             print("Missing values/NULL found")
      
      def replace_null(self):
          self.data.dropna(inplace=True)
          return self.data
      