from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size:int=256
    vocab_size:int=50257
    n_layer:int=6
    n_head:int=6
    n_embd:int=384
    rope_base:int=10000
    rope_max_len:int=1024
    pe_method:str="rope"
    scale_factor:float=1.0
    original_max_len:int=256
    alpha:float=1.0
    beta:float=32.0
    yarn_use_temp:bool=True