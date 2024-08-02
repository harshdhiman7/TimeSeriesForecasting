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
      
      def Normalize(self,data):
          self.data=data
          index=self.data.index
          cols=self.data.columns
          scaler=MinMaxScaler()
          self.data=scaler.fit_transform(self.data)
          return pd.DataFrame(self.data,columns=cols,index=index)
      


      