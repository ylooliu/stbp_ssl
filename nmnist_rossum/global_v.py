import torch
import torch.nn as nn
#
dtype = None
n_steps = None
devices = None
vth=None
tau=None
tau_s=None

def init():
    global dtype, devices, n_steps, tau_s,  vth, tau
    dtype = torch.float32
    devices = torch.device("cuda:0")

    n_steps = 25
    tau_s= 1.1
    tau=0.8
    vth=0.2
