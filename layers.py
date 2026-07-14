import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from RoPE import apply_rope

class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super(CausalSelfAttention,self).__init__()
        assert (config.n_embd%config.n_head)==0

        self.c_attn=nn.Linear(config.n_embd,3*config.n_embd)
        self.c_proj=nn.Linear(config.n_embd,config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT=1
        self.n_embd=config.n_embd
        self.n_head=config.n_head

        self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size)).view(1,1,config.block_size,config.block_size))

    def forward(self,x,cos,sin,kv_cache=None):
        B,T,C=x.size()
        qkv=self.c_attn(x)
        q,k,v=qkv.split(self.n_embd,dim=2)
        q=q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        k=k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v=v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)

        q=apply_rope(q,cos,sin)
        k=apply_rope(k,cos,sin)
        if kv_cache is not None:
            k_cached,v_cached=kv_cache
            k=torch.cat([k_cached,k],dim=2)
            v=torch.cat([v_cached,v],dim=2)
        new_kv_cache=(k,v)
        T_q=q.size(2)
        T_k=k.size(2)
        attn=(q@k.transpose(-2,-1))/math.sqrt(k.size(-1))

        if kv_cache is None:
            attn=attn.masked_fill(self.bias[:,:,:T_q,:T_k]==0,float("-inf"))

        attn=F.softmax(attn,dim=-1)
        y=attn@v
        y=y.transpose(1,2).contiguous().view(B,T,C)
        y=self.c_proj(y)
        return y,new_kv_cache



class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc=nn.Linear(config.n_embd,4*config.n_embd)
        self.c_proj=nn.Linear(4*config.n_embd,config.n_embd)
        self.gelu=nn.GELU(approximate="tanh")

    def forward(self,x):
        x=self.c_fc(x)
        x=self.gelu(x)
        x=self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln_1=nn.LayerNorm(config.n_embd)
        self.attn=CausalSelfAttention(config)
        self.ln_2=nn.LayerNorm(config.n_embd)
        self.mlp=MLP(config)

    def forward(self,x,cos,sin,kv_cache=None):
        y,new_kv_cache=self.attn(self.ln_1(x),cos,sin,kv_cache)
        x=x+y
        x=x+self.mlp(self.ln_2(x))
        return x,new_kv_cache

