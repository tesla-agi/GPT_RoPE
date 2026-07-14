import torch
import torch.nn as nn
import math

class ntk_aware(nn.Module):
    def __init__(self,d_head,max_len,scale_factor,base=10000.0):
        super(ntk_aware, self).__init__()

        base=base*(scale_factor**(d_head/(d_head-2)))
        theta=torch.exp(torch.arange(0,d_head,2,dtype=torch.float32)*(-math.log(base)/d_head))
        position=torch.arange(0,max_len,dtype=torch.float32).unsqueeze(1)
        angle=position*theta
        full_angles=torch.repeat_interleave(angle,2,dim=-1)
        sin_table=torch.sin(full_angles)
        cos_table=torch.cos(full_angles)

        self.register_buffer('cos_table',cos_table,persistent=False)
        self.register_buffer('sin_table',sin_table,persistent=False)

    def forward(self,start,end):
        return self.cos_table[start:end],self.sin_table[start:end]

