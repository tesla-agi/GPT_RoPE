import torch
import torch.nn as nn
import math

class YaRN(nn.Module):
    def __init__(self,d_head,max_len,original_max_len,scale_factor,alpha=1.0,beta=32.0,base=10000.0,use_temperature=True):
        super(YaRN,self).__init__()

        theta=torch.exp(torch.arange(0,d_head,2,dtype=torch.float32)*(-math.log(base)/d_head))
        cycles=(original_max_len*theta)/(2*math.pi)
        gamma=((cycles-alpha)/(beta-alpha)).clamp(0,1)
        new_theta=(1-gamma)*(theta/scale_factor)+gamma*theta
        position=torch.arange(0,max_len,dtype=torch.float32).unsqueeze(1)
        angle=position*new_theta
        full_angles=torch.repeat_interleave(angle,2,dim=-1)

        if use_temperature and scale_factor>1:
            inv_t=0.1*math.log(scale_factor)+1
            sqrt_inv_t=math.sqrt(inv_t)

        else:
            sqrt_inv_t=1.0

        cos_table=torch.cos(full_angles)*sqrt_inv_t
        sin_table=torch.sin(full_angles)*sqrt_inv_t

        self.register_buffer('cos_table',cos_table,persistent=False)
        self.register_buffer('sin_table',sin_table,persistent=False)

    def forward(self,start,end):
        return self.cos_table[start:end],self.sin_table[start:end]