import torch
import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F 

class SWD(nn.Module):
    """takes 2 input 
    """
    def __init__(self, 
                 w : int = 6,
                 encoder:bool = True,
                 decoder:bool = False,
                 T1:int = 512,
                 n:int = 3,# layers in encoder
                 m:int = 3,#layers in decoder
                 T2:int=512
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
            self.lin2 = nn.Linear(T1,T1)
            self.n = n 
        #SWD for decoder takes x1 as acoustic length T1 and x2 as text embedding of T2 length
        if self.decoder :
            self.lin1 = nn.Linear(T1,T1)
            self.lin2 = nn.Linear(T2,T2)
            self.m = m 
        self.w = w
                
    def forward(self,x1,x2,i=2,j=3):
        
        _,T1 = x1.shape
        _,T2 = x2.shape       

        if self.encoder and T1 !=T2 :
            raise Exception('for encoder, inputs are the same sequence')
        
        x1 = torch.transpose(self.lin1(x1),0,1)#T1,bs
        x2 = self.lin2(x2)#bs,T2

        tokenInteraction = torch.from_numpy((np.dot(x1.detach().numpy(),x2.detach().numpy())))# T1,T2

        # t2 is len of label seqeunce (x2)
        # t1 is len of encoder output (x1)
        if self.encoder and i is not None : 
            windSize = self.w+i 
        elif self.decoder and j is not None:
            windSize = self.w+j
        else:
            windSize = self.w
        
        # new mask generated each time 
        maskMat = torch.zeros((T2,T1))
        lenReqd = windSize//2 #initial window size w se ek 0,1 filled matrix 
        for i in range(lenReqd):
            torch.diagonal(maskMat,-i).fill_(1)
            torch.diagonal(maskMat,i).fill_(1)
      
        # sliding window of (w+i)
        e = windSize- self.w
        zer = torch.zeros(T1, self.w+e)
    
        ### when T1<T2 then problem arises as in the last there is no T1 left as window hence we can make it an overlapping kind of window 
        # and put in use T2/windseize but le T1 leng se rhe hai oth iliye agar T2 kam hota hai toh sirf T 2 krange jitna hi jate hai 
        # zayda hoteh ai toh bhi last ke khaali hai # same size wala shi shai hai sirf.
    
        for k in range(T2//windSize):
          
            SlidingWin = maskMat[:,(k)*(self.w+e):((k+1)*(self.w+e))]
            print(SlidingWin.shape)
            win = torch.from_numpy((np.dot(tokenInteraction.detach().numpy(),SlidingWin.detach().numpy())))

            zer = torch.cat((zer,win),dim=1)
    
        out = zer[:,self.w+e:]   
        print(out.shape,type(out),T1,T2,'======11111') 
        if out.shape[-1] < T2 :  
            out = F.pad(out,(T2-out.shape[-1],0))
        print(out.shape,type(out),T1,T2,'======')   
        
        localAtt = torch.transpose(out,0,1)
        print(localAtt.shape)
        
        gloabalAtt = torch.randn((T2,T1))##of shape T2,T1 
        print(gloabalAtt.shape)
        
        LGatt = torch.add(localAtt,gloabalAtt)
        print(LGatt.shape)##of shape T2,T1 
        
        m = nn.Softmax(dim=0)
        
        output = m(LGatt)
        print(output)
 
        
s = SWD(T1=9,T2=5,encoder = False, decoder = True,w=2)
from torchinfo import summary
summary(s,input_data=(torch.randn((5,9)),torch.randn((5,5))))
        