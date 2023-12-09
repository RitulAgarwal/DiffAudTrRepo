import torch 
import torch.nn as nn 
from scipy.special import softmax

import torch.nn.functional as F 
class Criterion(nn.Module):
    """
    mix of MSE and KL divergence loss 
    """
    def __init__(self) -> None:
        super().__init__()
        #self.ssim = SSIM(n_channels=1)
        self.MaE = nn.L1Loss()  
        self.Mse = nn.MSELoss()  
        self.kl_loss = nn.KLDivLoss(reduction="batchmean",log_target=True)    
        self.softmax = torch.nn.Softmax(dim = 2)

    def forward(self,input,target):
        #l1 = 1. - self.ssim(nn.functional.normalize(input).unsqueeze(1), nn.functional.normalize(target).unsqueeze(1))
        # l2 = nn.functional.kl_div(nn.functional.normalize(input).flatten(), nn.functional.normalize(target).flatten(),
        #                           reduction="batchmean", log_target=True)        
        l1 = self.Mse(input, target)  
        print(l1,'l1')
        # input = F.log_softmax(input)
        # input = torch.tensor(input,requires_grad=True)
        # input = self.softmax(input)
        # target = self.softmax(target)
        # l2 = self.kl_loss(F.log_softmax(input), F.softmax(target)) ### 0
        # probabilities = torch.tensor(softmax(input))
        
        
        # input = F.log_softmax(torch.tensor(input, requires_grad=True), dim=1)
        # # Sample a batch of distributions. Usually this would come from the dataset
        # target = F.softmax(torch.tensor(target), dim=1)
        
        # probabilities = F.softmax(input)
        
        # print(input.max(),input.min(),torch.sum(input))
        
        # print(target.max(),target.min(),torch.sum(target))
        
        
        # l2 = self.kl_loss(input,target)
    
        # # l2 = self.kl_loss(input, target)  #### nan
        # # print(F.log_softmax(input,dim=1).flatten().min(),F.log_softmax(input,dim=1).flatten().max(),torch.mean(F.log_softmax(input,dim=1).flatten()),torch.std(F.log_softmax(input,dim=1).flatten()))
        # # print(F.log_softmax(target,dim=1).flatten().min(),F.log_softmax(target,dim=1).flatten().max(),torch.mean(F.log_softmax(target,dim=1).flatten()),torch.std(F.log_softmax(target,dim=1).flatten()))
        # # l2 = self.kl_loss(input.flatten(),target.flatten())
        # print(l2,'this is l2')
        # # print(l1+l2) 
        return l1 #+ l2
    
# import torch
# loss = Criterion()
# input = torch.randn((16,80,400))
# output = torch.randn((16,80,400))

# print(loss(input,output))
