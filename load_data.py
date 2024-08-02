import numpy as np
import os
import pandas as pd

class DataLoader:
      def __init__(self,file_path):
          self.file_path=file_path

      def load_file(self):
          file_extension=os.path.splitext(self.file_path)[1]              
          if file_extension=='.csv':
             data=pd.read_csv(self.file_path)
          elif file_extension in ['.xls','.xlsx']:
              data=pd.read_excel(self.file_path)
          else:
              raise ValueError("Unsupported file type")       
          
          return data
      