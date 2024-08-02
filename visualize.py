import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Visualize:
      def __init__(self,data):
          self.data=data 
 
      def line_plot(self):
          plt.figure(figsize=(10,10),dpi=150)
          cols=self.data.columns
          for i in range(self.data.shape[1]):
                plt.subplot(1,self.data.shape[1],i+1)
                plt.plot(self.data[cols[i]],color='blue')
          return plt.show()        