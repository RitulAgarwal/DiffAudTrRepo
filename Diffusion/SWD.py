import torch
import numpy as np 
import torch.nn as nn 

class SWD(nn.Module):
    """takes 2 input 
    """
    def __init__(self, 
                 w : int = 6,
                 encoder:bool = False,
                 decoder:bool = True,
                 T1:int = 512,
                 n:int = 3,# layers in encoder
                 m:int = 3,#layers in decoder
                 T2:int=512,
                 emb:int=30
                 ) -> None:
        
        super().__init__()
        self.T1 = T1 
        self.T2 = T2 
        self.encoder = encoder
        self.decoder = decoder
        if self.encoder == self.decoder :
            raise Exception('either compute for encoder or decoder.')
        #SWD for encoder takes same x1,x2 as acoustic sequence (T1 length)
        if self.encoder :
            self.lin1 = nn.Linear(T1,T1)
            self.lin2 = nn.Linear(T1,emb)
            self.n = n 
        #SWD for decoder takes x1 as acoustic length T1 and x2 as text embedding of T2 length
        if self.decoder :
            self.lin1 = nn.Linear(T1,emb)
            self.lin2 = nn.Linear(T2,emb)
            self.m = m 
        self.w = w
                
    def forward(self,x1,x2,i=None,j=None):
        _,T1 = x1.shape
        _,T2 = x2.shape       
        # print(x1.shape,x2.shape,'') # 5,1024 and 5,768   
        x1 = self.lin1(x1)#Bs,dim
        x2 = torch.transpose(self.lin2(x2),0,1)#dim,bs
        dotted = torch.from_numpy((np.dot(x1.detach().numpy(),x2.detach().numpy())))# BS,Bs
        # t2 is len of label seqeunce (x2)
        # t1 is len of encoder output (x1)
        maskMat = torch.zeros((T2,T1))
        lenReqd = self.w//2 #initial window size w se ek 0,1 filled matrix 
        for i in range(lenReqd):
            torch.diagonal(maskMat,-i).fill_(1)
            torch.diagonal(maskMat,i).fill_(1)
        print(maskMat.shape)
        
        #mask is (w+i)
        if self.encoder and i is not None : 
            windSize = self.w+i 
        elif self.decoder and j is not None:
            windSize = self.w+j
        else:
            windSize = self.w
            
        SldingWindow = maskMat[:windSize,:]
        print(SldingWindow.shape,'*****************************************')
        
        localAtt = torch.from_numpy((np.dot(maskMat.detach().numpy(),dotted.detach().numpy())))##of shape T2,T1 
    
        gloabalAtt = torch.randn((T2,T1))##of shape T2,T1 
        
        LGatt = torch.add(localAtt,gloabalAtt)
        print(LGatt)##of shape T2,T1 
        
        m = nn.Softmax(dim=0)
        
        output = m(LGatt)
        print(output)
        
                
        
        
        
        
        
              
            
        
s = SWD(T1=1024,T2=768)
from torchinfo import summary
summary(s,input_data=(torch.randn((5,1024)),torch.randn((5,768))))
        