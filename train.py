import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.optim as optim 

class ScaledDotProductAttention(nn.Module):
      def __init__(self,d_k):
          super(ScaledDotProductAttention,self).__init__()  
          self.d_k=d_k 

       def forward(self,Q,K,V,mask=None):
           scores=torch.matmul(Q,K.transpose(-2,-1))/ np.sqrt(self.d_k)
           if mask is not None:
              scores = scores.masked_fill(mask == 0, -1e9)
           attn=torch.nn.Softmax(scores,dim=-1)
           output=torch.matmul(attn,V)

           return output,attn 
          