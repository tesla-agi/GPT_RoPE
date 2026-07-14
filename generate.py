import torch
import torch.nn.functional as F
import time
import tiktoken
from model import GPT
from gpt_config import GPTConfig


@torch.no_grad()
def generate(
        model,
        idx,
        max_new_tokens,
        strategy="greedy",
        temperature=1.0,
        top_k=None,
        top_p=None,):

    model.eval()


    logits,_,kv_caches=model(idx,kv_cache=None)
    logits=logits[:,-1,:]
    for _ in range(max_new_tokens):
        if strategy=="greedy":
            next_token=torch.argmax(logits,dim=-1,keepdim=True)

        elif strategy=="temperature":
            logits=logits/temperature
            probs=F.softmax(logits,dim=-1)
            next_token=torch.multinomial(probs,1)

        elif strategy=="top_k":
            logits=logits/temperature
            topk_values,topk_indices=torch.topk(logits,k=top_k,dim=-1)
            threshold=topk_values[:,[-1]]
            logits[logits<threshold]=float("-inf")
            probs=F.softmax(logits,dim=-1)
            next_token=torch.multinomial(probs,1)

        elif strategy=="top_p":
            logits=logits/temperature
            probs=F.softmax(logits,dim=-1)
            sorted_probs,sorted_indices=torch.sort(probs,dim=-1,descending=True)
            cumulative_probs=torch.cumsum(sorted_probs,dim=-1)
            sorted_indices_to_remove=cumulative_probs>top_p
            sorted_indices_to_remove[:,1:]=sorted_indices_to_remove[:,:-1].clone()
            sorted_indices_to_remove[:,0]=False
            sorted_probs[sorted_indices_to_remove]=0
            sorted_probs=sorted_probs/sorted_probs.sum(dim=-1,keepdim=True)
            sample_idx=torch.multinomial(sorted_probs,1)
            next_token=sorted_indices.gather(-1,sample_idx)


        idx=torch.cat((idx,next_token),dim=1)
        logits,_,kv_caches=model(next_token,kv_cache=kv_caches)
        logits=logits[:,-1,:]

    return idx

# Setup
device="cpu"
if torch.cuda.is_available():
    device="cuda"
elif hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
    device="mps"
print(f"Using device: {device}")


# Encode prompt
enc=tiktoken.get_encoding("gpt2")
prompt="To be, or not to be"
tokens=enc.encode(prompt)
idx=torch.tensor(tokens,dtype=torch.long,device=device).unsqueeze(0)

# Time it
torch.manual_seed(42)
strategies=[
    ("greedy", {}),
    ("temperature", {"temperature": 0.8}),
    ("top_k", {"top_k": 40, "temperature": 1.0}),
    ("top_p", {"top_p": 0.9, "temperature": 1.0}),
]



pe_methods = ["rope","pi","ntk_aware","ntk_by_parts","yarn"]

for pe_method in pe_methods:
    print(f"\n{'='*60}\n>>> Running pe_method='{pe_method}',scale_factor=2.0\n{'='*60}")
    cfg=GPTConfig(pe_method=pe_method, scale_factor=2.0)
    model=GPT(cfg)
    state=torch.load("rope_gpt2.pth",map_location=device)
    state={k:v for k, v in state.items() if not (k.endswith("cos_table") or k.endswith("sin_table"))}
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    torch.manual_seed(42)
    with open(f"generations_{pe_method}.txt","w") as f:
        for strategy, kwargs in strategies:
            print(f"\n=== {strategy}{kwargs} ===")
            start=time.time()
            out=generate(model,idx.clone(),max_new_tokens=500,strategy=strategy,**kwargs)
            elapsed=time.time()-start
            f.write(f"=== {strategy} {kwargs} ===\n")
            f.write(enc.decode(out[0].tolist()) + "\n\n")
            print(f"time elapsed:{elapsed:.2f}s")
            print(enc.decode(out[0].tolist()))

