import random
import math

def difficulty(t): return math.log1p(len(t))

def build_curriculum(texts):
    texts=sorted(texts,key=difficulty)
    n=len(texts)
    return {
        "easy":texts[:n//3],
        "medium":texts[n//3:2*n//3],
        "hard":texts[2*n//3:]
    }

class Scheduler:
    def __init__(self,c): self.c=c
    def get_stage(self,step,total):
        p=step/total
        return "easy" if p<0.33 else "medium" if p<0.66 else "hard"
    def sample(self,stage):
        return random.choice(self.c[stage])
