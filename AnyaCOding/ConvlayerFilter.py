import torch 
import torch.nn as nn 

# from diffusers import AudioLDMPipeline
# repo_id = "cvssp/audioldm-s-full-v2"
# ldm_pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
# ldm_unet = ldm_pipe.unet
# print(ldm_unet) # 8 channels ka input conv2d hai 
# ldm_textEncoder = ldm_pipe.text_encoder
# print(ldm_textEncoder)
import matplotlib.pyplot as plt

# class nn_test(nn.Module):
#     def __init__(self,
#                  input_channels = 1,
#                  output_channels = 4,
#             ):
        
#         super(nn_test ,self).__init__()

#         self.LAYER0 = nn.Conv1d(in_channels=3, out_channels=10, kernel_size=3)
#         self.LAYER1 = nn.Linear(5, 10)
#     def forward(self,o):
        
#         o = self.LAYER0(o)
        
#         return o 

LAYER0 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3)
LAYER1 = nn.Linear(5, 10)
LAYER2 = nn.Conv1d(in_channels=4, out_channels=10, kernel_size=3)

# net = nn_test()
a = torch.randn(1,1,500)
print(a.shape)
b = LAYER0(a)
print(b.shape)
c = LAYER2(b)
print(c.shape)

# f,plt_arr = plt.subplots(3,sharex=True)
# f.suptitle('compare ')
# plt_arr[0].plot(a)
# plt_arr[0].set_title('original')
# plt_arr[1].plot(b.detach().numpy())
# plt_arr[1].set_title('after conv')
# plt_arr[2].plot(c.detach().numpy())
# plt_arr[2].set_title('after conv1')
# plt.savefig('what_to_data')
# First Scatter plot
plt.scatter(b.squeeze()[3].detach().numpy(), range(len(b.squeeze()[0])))
plt.savefig('origianldata4')
# for ind,i in enumerate(b) :
#     plt.scatter(i[-1].detach().numpy(),i[-2].detach().numpy())
#     plt.savefig('origianldata'+ str(ind))

# #Second Scatter plot
# plt.scatter(b[0].detach().numpy(), b[1].detach().numpy(), c ="k",linewidths = 2,marker ="p",edgecolor ="red",s = 150,alpha=0.5)

# plt.scatter(c[0].detach().numpy(), c[1].detach().numpy(), c ="r",linewidths = 2, marker ="D", edgecolor ="g", s = 70, alpha=0.5)

# plt.title('Multiple Scatter plot')
plt.xlabel('x-axis')
plt.ylabel('y-axis')


# plt_arr[0].plot(a_dash)
# plt_arr[0].set_title('original')
# plt_arr[1].plot(b_dash.detach().numpy())
# plt_arr[1].set_title('after linear')
