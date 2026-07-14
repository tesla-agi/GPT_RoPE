import torch
from model import GPT
from gpt_config import GPTConfig
from data import DataLoaderLite
import math
import time

def train_GPT():
    cfg=GPTConfig()
    device="mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(42)
    train_loader=DataLoaderLite(B=16,T=cfg.block_size)
    model=GPT(cfg).to(device)
    max_lr=3e-4
    min_lr=max_lr*0.1
    warmup_steps=100
    max_steps=5000

    def get_lr(step):
        if step<warmup_steps:
            return max_lr*(step+1)/warmup_steps
        if step>=max_steps:
            return min_lr
        decay_ratio=(step-warmup_steps)/(max_steps-warmup_steps)
        coefficient=0.5*(1.0+math.cos(math.pi*decay_ratio))
        return min_lr+coefficient*(max_lr-min_lr)

    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(0.9,0.95),weight_decay=0.1)
    for i in range(max_steps):
        lr=get_lr(i)
        for param_group in optimizer.param_groups:
            param_group['lr']=lr

        x,y=train_loader.next_batch()
        x,y=x.to(device),y.to(device)

        t0=time.time()

        optimizer.zero_grad()
        logits,loss,_=model(x,y)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()

        if device=="mps":
            torch.mps.synchronize()

        dt=time.time()-t0
        tok_per_sec=(train_loader.B*train_loader.T)/dt
        if i%50==0 or i==max_steps-1:
            print(f"step: {i:4d} | loss: {loss.item():.4f} | lr: {lr:.2e} | "
              f"grad_norm: {norm:.2f} | dt: {dt*1000:.1f}ms | tok/s: {tok_per_sec:.0f}")

        if i>0 and i%500==0:
            torch.save(model.state_dict(),"rope_gpt2.pth")
            print(f"Saved checkpoint at step {i}")

    torch.save(model.state_dict(),"rope_gpt2.pth")
    print("Saved model weights to rope_gpt2.pth")


if __name__ == "__main__":
    train_GPT()