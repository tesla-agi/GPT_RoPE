import torch
import torch.nn as nn
import math


class RoPE(nn.Module):
    def __init__(self,d_head,max_len,base=10000.0):
        super().__init__()

        theta=torch.exp(torch.arange(0,d_head,2,dtype=torch.float32)*(-math.log(base)/d_head))
        position=torch.arange(0,max_len,dtype=torch.float32).unsqueeze(1)
        angles=position*theta
        full_angles=torch.repeat_interleave(angles,2,dim=-1)
        sin_table=torch.sin(full_angles)
        cos_table=torch.cos(full_angles)

        self.register_buffer('sin_table',sin_table,persistent=False)
        self.register_buffer('cos_table',cos_table,persistent=False)

    def forward(self,start,end):
        return self.cos_table[start:end],self.sin_table[start:end]


def rotate_half(x):
    x=x.reshape(*x.shape[:-1],-1,2)
    x=x[...,[1,0]]
    x=x*torch.tensor([-1,1],device=x.device)
    x=x.flatten(-2)
    return x

def apply_rope(x,cos,sin):
    return x*cos+rotate_half(x)*sin







