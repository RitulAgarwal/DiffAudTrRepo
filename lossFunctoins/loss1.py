import torch.nn as nn 

class Criterion(nn.Module):
    """
    mix of MSE and KL divergence loss 
    """
    def __init__(self) -> None:
        super().__init__()
        #self.ssim = SSIM(n_channels=1)
        self.MSE = nn.MSELoss()        
    
    def forward(self,input,target):
        #l1 = 1. - self.ssim(nn.functional.normalize(input).unsqueeze(1), nn.functional.normalize(target).unsqueeze(1))
        l2 = nn.functional.kl_div(nn.functional.normalize(input).flatten(), nn.functional.normalize(target).flatten(),
                                  reduction="batchmean", log_target=True)
        l1 = self.MSE(input, target)  
        print(l1+l2) 
        return l1 + l2