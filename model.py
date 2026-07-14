import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import Block
from RoPE import RoPE
from PI import PI
from ntk_aware import ntk_aware
from ntk_bp import NTKByParts
from YaRN import YaRN

class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config

        self.transformer=nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size,config.n_embd),
                h=nn.ModuleList([
                    Block(config) for _ in range(config.n_layer)
                ]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head=nn.Linear(config.n_embd,config.vocab_size,bias=False)
        self.transformer.wte.weight=self.lm_head.weight
        d_head=config.n_embd//config.n_head
        if config.pe_method=="rope":
            self.rope=RoPE(d_head=d_head,max_len=config.rope_max_len,base=config.rope_base)
        elif config.pe_method=="pi":
            self.rope=PI(d_head=d_head,max_len=config.rope_max_len,scale_factor=config.scale_factor,base=config.rope_base)
        elif config.pe_method=="ntk_aware":
            self.rope=ntk_aware(d_head=d_head,max_len=config.rope_max_len,scale_factor=config.scale_factor,base=config.rope_base)
        elif config.pe_method=="ntk_by_parts":
            self.rope=NTKByParts(d_head=d_head,scale_factor=config.scale_factor,max_len=config.rope_max_len,original_max_len=config.original_max_len,
                                 alpha=config.alpha,beta=config.beta,base=config.rope_base)
        elif config.pe_method=="yarn":
            self.rope=YaRN(d_head=d_head,max_len=config.rope_max_len,original_max_len=config.original_max_len,scale_factor=config.scale_factor,
                           alpha=config.alpha,beta=config.beta,base=config.rope_base,use_temperature=config.yarn_use_temp)

        else:
            raise ValueError(f"Unknown pe_method: {config.pe_method}")
        self.apply(self._init_weights)

    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            std=0.02
            if hasattr(module,'NANOGPT_SCALE_INIT'):
                std*=(2*self.config.n_layer)**-0.5
            torch.nn.init.normal_(module.weight,mean=0.0,std=std)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)

    def forward(self,idx,targets=None,kv_cache=None):
        B,T=idx.size()
        if kv_cache is not None and kv_cache[0] is not None:
            past_length=kv_cache[0][0].size(2)

        else:
            past_length=0

        assert past_length+T<=self.config.rope_max_len, \
            f"Total length {past_length + T} exceeds rope_max_len {self.config.rope_max_len}"

        token_emb=self.transformer.wte(idx)
        x=token_emb

        if kv_cache is None:
            kv_cache=[None]*len(self.transformer.h)

        new_kv_cache=[]
        start=past_length
        end=past_length+T
        cos,sin=self.rope(start,end)
        for i,block in enumerate(self.transformer.h):
            x,new_cache=block(x,cos=cos,sin=sin,kv_cache=kv_cache[i])
            new_kv_cache.append(new_cache)

        x=self.transformer.ln_f(x)
        logits=self.lm_head(x)
        loss=None
        if targets is not None:
            loss=F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))

        return logits,loss,new_kv_cache
