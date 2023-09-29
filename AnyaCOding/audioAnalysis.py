import torchaudio 
import torch.nn as nn 
import matplotlib.pyplot as plt 

audio,sr = torchaudio.load('/home/earth/DATASETS/DATASET_BPCC_AUDIO/English/comparable_0.wav')
print(audio.shape,sr)
layer1 = nn.Conv1d(1,4,11,dilation=1)
layer2 = nn.Conv1d(1,4,7,dilation=2)
layer3 = nn.Conv1d(1,4,5,dilation=1)
layer4 = nn.Conv1d(1,4,3,dilation=2)
audio_re = layer1(audio)
print(audio_re.shape)


f,plt_arr = plt.subplots(5,sharex=True)
f.suptitle('compare')

plt_arr[0].plot(audio[0].detach().numpy())
# plt_arr[0].set_title('original')
torchaudio.save('AudiosAnalys/11DIL1ORIGaudio.wav',audio,sr)
plt_arr[1].plot(audio_re[0].detach().numpy())
# plt_arr[1].set_title('1st')
torchaudio.save('AudiosAnalys/11DIL11staudio.wav',audio_re[0].unsqueeze(dim=0),sr)
plt_arr[2].plot(audio_re[1].detach().numpy())
# plt_arr[2].set_title('2nd')
torchaudio.save('AudiosAnalys/11DIL12ndaudio.wav',audio_re[1].unsqueeze(dim=0),sr)
plt_arr[3].plot(audio_re[2].detach().numpy())
# plt_arr[3].set_title('3rd')
torchaudio.save('AudiosAnalys/11DIL13rdaudio.wav',audio_re[2].unsqueeze(dim=0),sr)
plt_arr[4].plot(audio_re[3].detach().numpy())
# plt_arr[4].set_title('4th')
torchaudio.save('AudiosAnalys/11DIL14thaudio.wav',audio_re[3].unsqueeze(dim=0),sr)

plt.savefig('AudiosAnalys/11DIL1AUDIOS.png')
# First Scatter plot
# plt.scatter(b.squeeze()[3].detach().numpy(), range(len(b.squeeze()[0])))
# plt.savefig('origianldata4')
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


