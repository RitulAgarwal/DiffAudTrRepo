# from collections import OrderedDict
# import torch
# from torch import nn, optim
# from ignite.engine import *
# from ignite.handlers import *
# from ignite.metrics import *
# from ignite.utils import *
# from ignite.contrib.metrics.regression import *
# from ignite.contrib.metrics import *

# def eval_step(engine, batch):
#     return batch
# default_evaluator = Engine(eval_step)
# param_tensor = torch.zeros([1], requires_grad=True)
# default_optimizer = torch.optim.SGD([param_tensor], lr=0.1)
# def get_default_trainer():
#     def train_step(engine, batch):
#         return batch
#     return Engine(train_step)
# default_model = nn.Sequential(OrderedDict([
#     ('base', nn.Linear(4, 2)),
#     ('fc', nn.Linear(2, 1))
# ]))
# metric = SSIM(data_range=1.0)
# metric.attach(default_evaluator, 'ssim')
# preds = torch.rand((5,1,80,883))
# target = preds * 0.75
# state = default_evaluator.run([[preds, target]])
# print(state.metrics['ssim'])



import torch
# from pytorch_msssim import ssim,SSIM, MS_SSIM,ms_ssim

# # # reuse the gaussian kernel with SSIM & MS_SSIM. 
# # ssim_module = SSIM(data_range=255, size_average=True, channel=1) # channel=1 for grayscale images
# # ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=1)

# # ssim_loss = 1 - ssim_module(torch.randn((5,1,161,883)), torch.randn((5,1,161,883)))
# # ms_ssim_loss = 1 - ms_ssim_module(torch.randn((5,1,161,883)), torch.randn((5,1,161,883)))

# # print(ssim_loss,ms_ssim_loss)

# ssim_loss = 1 - ssim( torch.randn((5,1,161,883)), torch.randn((5,1,161,883)), data_range=255, size_average=True) # return a scalar
# ms_ssim_loss = 1 - ms_ssim( torch.randn((5,1,161,883)), torch.randn((5,1,161,883)), data_range=255, size_average=True )

# print(ssim_loss,ms_ssim_loss)










from piqa import SSIM

# # class SSIMLoss(SSIM):
# #     def forward(self, x, y):
# #         return 1. - super().forward(x, y)

# criterion = SSIMLoss() # .cuda() if you need GPU support
# # print(criterion)
# # loss = criterion(torch.rand((5,3,80,883)), torch.rand((5,3,80,883)))

# # print(loss)
# from torchinfo import summary
criterion = SSIM(n_channels=1) # .cuda() if you need GPU support

# summary(criterion, depth=8)
# print(dir(criterion))
# print(SSIM.named_parameters())
loss = criterion(torch.rand((5,1,80,883)), torch.rand((5,1,80,883)))

print(1. - loss)








# import pytorch_ssim
# import torch
# from torch.autograd import Variable

# img1 = torch.rand((5, 1, 80, 883))
# img2 = torch.rand((5, 1, 80, 883))

# if torch.cuda.is_available():
#     img1 = img1.cuda()
#     img2 = img2.cuda()

# # print(pytorch_ssim.ssim(img1, img2))

# ssim_loss = pytorch_ssim.SSIM(window_size = 11)

# print(ssim_loss(img1, img2))




















import torch.nn.functional as F
import torch.nn as nn 
import torch 

#log_tarhet being set to True or False determines whether the target distribution is provided in a logarithmic form or not.
#log_target" is set to False (the default), it means that the target distribution is provided in a non-logarithmic form. In this case, the KL divergence loss will internally compute the logarithm of the target distribution to calculate the divergence.
#mel Spec are mel scaled so we dont want to compute log or do we ? 
# kl_loss =nn.functional.kl_div(reduction="batchmean", log_target=True)
# kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
layer = nn.Flatten()
# input = layer(torch.randn((5,1,80,883)))
# print(input.shape)
# log_t = layer(torch.randn((5,1,80,883)))
input = torch.nn.functional.normalize(layer(torch.randn((5,1,80,883))), p=2.0)
log_t = torch.nn.functional.normalize(layer(torch.randn((5,1,80,883))), p=2.0)
output = nn.functional.kl_div(input,log_t,reduction="batchmean", log_target=True)
print(output,'******************')








