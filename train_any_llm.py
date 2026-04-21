import torch

class Trainer:
    def __init__(self,model,opt,tokenizer):
        self.model=model
        self.opt=opt
        self.tok=tokenizer

    def encode(self,t):
        return torch.tensor(self.tok.encode(t).ids)

    def step(self,x,y):
        _,loss=self.model(x,y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()

def train(model,trainer,scheduler,steps=200):
    for i in range(steps):
        stage=scheduler.get_stage(i,steps)
        text=scheduler.sample(stage)
        t=trainer.encode(text)
        if len(t)<2: continue
        x=t[:-1].unsqueeze(0)
        y=t[1:].unsqueeze(0)
        loss=trainer.step(x,y)
        if i%25==0:
            print(i,stage,loss)
