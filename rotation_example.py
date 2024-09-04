#!/usr/bin/env python
# coding: utf-8

# In[1]:


import icon_registration as icon
import icon_registration.data
import icon_registration.networks as networks
from icon_registration.config import device

import numpy as np
import torch
import torchvision.utils
import matplotlib.pyplot as plt


# In[101]:


ds, _ = icon_registration.data.get_dataset_mnist(split="train", number=9)

sample_batch = next(iter(ds))[0]
plt.imshow(torchvision.utils.make_grid(sample_batch[:12], nrow=4)[0])


# In[102]:


inner_net = icon.FunctionFromVectorField(networks.tallUNet2(dimension=2))

for _ in range(3):
     inner_net = icon.TwoStepRegistration(
         icon.DownsampleRegistration(inner_net, dimension=2),
         icon.FunctionFromVectorField(networks.tallUNet2(dimension=2))
     )

net = icon.GradientICON(inner_net, icon.LNCC(sigma=4), lmbda=.5)
net.assign_identity_map(sample_batch.shape)


# In[103]:


net.train()
net.to(device)

optim = torch.optim.Adam(net.parameters(), lr=0.001)
curves = icon.train_datasets(net, optim, ds, ds, epochs=5)
plt.close()
plt.plot(np.array(curves)[:, :3])


# In[104]:


class RotationNet(icon.RegistrationModule):
    def __init__(self):
        super().__init__()
        self.angle = torch.nn.Parameter(torch.Tensor([0.]).float())
    def forward(self, A, B):
        def warp(coords):
            coords = coords - .5
            output = [
                coords[:, :1] * torch.cos(self.angle[0]) + coords[:, 1:] * torch.sin(self.angle[0]),
               -coords[:, :1] * torch.sin(self.angle[0]) + coords[:, 1:] * torch.cos(self.angle[0])
            ]
            output = torch.cat(output, axis=1)
            return output + .5
        return warp
    def set_angle(self, theta):
        with torch.no_grad():
            self.angle = torch.nn.Parameter(torch.Tensor([theta]).float())
            


# In[105]:


rot = RotationNet()
rot.set_angle(3)


# In[ ]:





# In[114]:


def optim(net, params, A, B):
    net.assign_identity_map(sample_batch.shape)
    net.cuda()
    A = A.cuda() 
    B = B.cuda()
    o = torch.optim.Adam(params, lr=0.02)
    for i in range(8):
        for j in range(10):
            o.zero_grad()
            loss = net(A, B)
            
            loss.all_loss.backward()
            o.step()


# In[115]:


A = sample_batch[:1]
B = sample_batch[1:2]


# In[ ]:





# In[ ]:





# In[116]:


angles = []
losses = []
for i in range(40):
    angle = i / 40 * 2 * np.pi
    rot.set_angle(angle)

    stage_net = icon.GradientICON(icon.TwoStepRegistration(rot, inner_net), icon.LNCC(sigma=4), lmbda=0)
    stage_net.assign_identity_map(sample_batch.shape)
    stage_net.cuda()
    optim(stage_net, rot.parameters(), A, B)
    angles.append(rot.angle.item())
    losses.append(stage_net(A.cuda(), B.cuda()).all_loss.item())
plt.plot(losses)
plt.show()
plt.plot(angles)
plt.show()


# In[117]:


angles = []
losses = []
for i in range(40):
    angle = i / 40 * 2 * np.pi
    rot.set_angle(angle)

    stage_net = icon.GradientICON(rot, icon.LNCC(sigma=4), lmbda=0)
    stage_net.assign_identity_map(sample_batch.shape)
    stage_net.cuda()
    optim(stage_net, rot.parameters(), A, B)
    angles.append(rot.angle.item())
    losses.append(stage_net(A.cuda(), B.cuda()).all_loss.item())
plt.plot(losses)
plt.show()
plt.plot(angles)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




