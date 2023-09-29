import torch
import torch.nn as nn
from torchinfo import summary
import numpy as np
import torch.nn.functional as F 

#(5005, 9413) (44, 2503) torch.Size([1, 80, 614])
#(4935, 7136) (42, 3763) torch.Size([1, 80, 645])
#(2051, 5531) (44, 2315) torch.Size([1, 80, 681])
#(7274, 9267) (36, 1406) torch.Size([1, 80, 687])
#(2558, 7888) (44, 3127) torch.Size([1, 80, 670])
#(4659, 6462) (44, 2611) torch.Size([1, 80, 622])
#(3455, 7994) (41, 3927) torch.Size([1, 80, 697])

class MultiRes(nn.Module):
    def __init__(self,
                 n_conv,
                 n_out_channels,
                 in_channel=1,
                 kernel_size_range=(3, 103),
                 stride_range=(1,15)):
        super().__init__()

        if n_out_channels % n_conv != 0:
            raise Exception("Give Correct parameters")
        
        self.kernels = np.sort(np.linspace(kernel_size_range[0], kernel_size_range[1], n_conv, dtype=np.int32))[::-1]
        self.strides = np.sort(np.linspace(stride_range[0], stride_range[1], n_conv, dtype=np.int32))[::-1]

        self.ConvList = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=in_channel, out_channels=int(n_out_channels/n_conv), kernel_size=self.kernels[i], stride=self.strides[i]),
                nn.ReLU(inplace=True)) for i in range(n_conv)])    
                
    def forward(self,x):
        out_list = [self.ConvList[i](x) for i in range(len(self.ConvList))]
        a = out_list[-1].shape[-1]
        for i in range(len(out_list)):
            out_list[i] = F.pad(out_list[i],(0,a-out_list[i].shape[-1]))
        return torch.cat(out_list, dim=1)   

# m = MultiRes(n_conv=10, n_out_channels=80,kernel_size_range=(4277,7670),stride_range=(42,2807))
# print(m.forward(x=(torch.randn(1,1,30000))).shape)# summary(m, (1,1,32000))
    
# iteration_num = 10000
# for i in range(iteration_num):
#     kernel_size1 = np.random.randint(3,9600)
#     kernel_size2 = np.random.randint(3,9600)
#     stride1 = np.random.randint(1,4000)
#     stride2 = np.random.randint(1,4000)
#     ks_tuple = tuple(sorted((kernel_size1,kernel_size2))) 
#     stride_tuple = tuple(sorted((stride1,stride2))) 
#     m = MultiRes(n_conv=10, n_out_channels=80,kernel_size_range=ks_tuple,stride_range=stride_tuple)
#     x = torch.randn(1,1,32000)
#     output = m.forward(x)
#     print(ks_tuple,stride_tuple,output.shape)
#     if output.shape[-1] in range(600,700):
#         break
        
#     del m,x,output
    
    
input = torch.randn(1,1,30)
filter1 = torch.randn(4,1,7)
filter2 = torch.randn(4,1,5)
filter3 = torch.randn(4,1,11)

output = F.conv1d(input,filter1)
print(output.shape)
output = F.conv1d(input,filter2)
print(output.shape)
output = F.conv1d(input,filter3)
print(output.shape)

