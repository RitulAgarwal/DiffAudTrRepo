import torch.nn as nn  

class SmallArch(nn.Module):
    def __init__(self):
        super(SmallArch ,self).__init__()
        self.LAYER0 = nn.Conv1d(in_channels=80, out_channels=20, kernel_size=3)
        self.relu = nn.ReLU()
        self.LAYER1 = nn.Conv1d(in_channels=20, out_channels=80, kernel_size=3,padding = 2 )
        
    def forward(self,x):
        return self.LAYER1(self.relu(self.LAYER0(x)))


