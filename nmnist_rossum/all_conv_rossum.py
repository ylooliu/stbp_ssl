import torch
import torch.nn as nn
import numpy as np
# from spikingjelly.clock_driven import encoding
from spikingjelly.clock_driven import neuron, functional
import global_v as glv

#cnn_layer(in_planes, out_planes, stride, padding, kernel_size)
conv1=[2,12,1,1,3]
conv2=[12,64,1,1,3]

linear1=[64*8*8, 10]


scale=1

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def topsp(input,pre_psp=None):
    tau_s = glv.tau_s

    if pre_psp is None:
        pre_psp=torch.zeros(input.shape).cuda()

    syn= pre_psp - pre_psp / tau_s + input
    syns = syn / tau_s

    return syns

class SpikeAct(torch.autograd.Function):
    """
        Implementation of the spiking activation function with an approximation of gradient.
    """

    @staticmethod
    def forward(ctx, input):
        Vth=glv.vth
        ctx.save_for_backward(input)
        # if input = u > Vth then output = 1
        output = torch.gt(input, Vth)
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # hu is an approximate func of df/du
        # hu = abs(input) < 0.5
        # hu = hu.float() / (2 * 0.5)

        grad_input=grad_input / (scale * torch.abs(input - glv.vth) + 1.) ** 2

        return grad_input


class LIFSpike(nn.Module):
    """
        Generates spikes based on LIF module. It can be considered as an activation function and is used similar to ReLU. The input tensor needs to have an additional time dimension, which in this case is on the last dimension of the data.
    """

    def __init__(self):
        super(LIFSpike, self).__init__()

    def forward(self, conv_psp, mem=None, spike=None):  #输入的 x是conv(psp)

        if mem==None and spike==None:
            mem =torch.zeros(conv_psp.shape, device=conv_psp.device)
            spike = torch.zeros(conv_psp.shape, device=conv_psp.device)

        mem, spike = self.state_update(mem, spike, conv_psp)
        return spike,mem

    def state_update(self, u_t_n1, o_t_n1, W_mul_o_t1_n ):
        tau=glv.tau
        u_t1_n1 = tau * u_t_n1 * (1 - o_t_n1) + W_mul_o_t1_n
        o_t1_n1 = SpikeAct.apply(u_t1_n1)
        return u_t1_n1, o_t1_n1




class SCNN(nn.Module):
    def __init__(self):
        super(SCNN, self).__init__()
        self.thresh = glv.vth
        self.decay = glv.tau

        in_planes, out_planes, stride, padding, kernel_size = conv1
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                               padding=padding)  #3-96 [32 32]

        self.dropout1 = nn.Dropout2d(0.1)
        self.maxpool1=nn.MaxPool2d(2)

        in_planes, out_planes, stride, padding, kernel_size = conv2
        self.conv2 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                               padding=padding)  #96-256  [32 32]


        self.maxpool1=nn.MaxPool2d(2)   #256  [16 16]



        in_planes, out_planes = linear1
        self.linear1 = nn.Linear(in_features=in_planes, out_features=out_planes)  #1024

    def layer_forward(self,inspike,psp,mem,spike,ops,dropout=None,maxpool=None):
        psp=topsp(inspike,psp)

        conv_psp=ops(psp)
        if dropout is not None:
            conv_psp=dropout(conv_psp)
        if maxpool is not None:
            conv_psp=maxpool(conv_psp)
        spike,mem=LIFSpike()(conv_psp, mem, spike)



        return psp,mem,spike

    def forward(self,x):
        spikes=[]
        mems=[]
        psp1 = psp2  = psp_out =None
        mem1=mem2=mem_out=None
        spike1=spike2=spike_out=None

        for t in range(glv.n_steps):
            psp1,mem1,spike1=self.layer_forward(x[...,t],psp1,mem1,spike1,self.conv1,maxpool=self.maxpool1)
            psp2, mem2, spike2 = self.layer_forward(spike1, psp2, mem2, spike2 ,self.conv2, maxpool=self.maxpool1)


            psp_out, mem_out, spike_out = self.layer_forward(torch.flatten(spike2,1,3), psp_out, mem_out, spike_out, self.linear1)

            mems.append(mem_out)
            spikes.append(spike_out)

        mems=torch.stack(mems,2)
        spikes=torch.stack(spikes,2)

        return mems,spikes
