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
      
      def split_data(self,data,train_date,test_date):
          self.data=data
          self.train_idx=train_date
          self.test_idx=test_date
          self.train_input=self.data.loc[:self.train_idx,['Open','High','Low','Close']]
          self.test_input=self.data.loc[self.test_idx:,['Open','High','Low','Close']]
          self.train_output=self.data.loc[:self.train_idx,['Volume']]
          self.test_output=self.data.loc[self.test_idx:,['Volume']]
          
          return {"train": {"input":self.train_input,"output":self.train_output},
                  "test":{"input":self.test_input,"output":self.test_output}}


      