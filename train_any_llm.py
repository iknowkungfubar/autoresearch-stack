"""
Training abstraction for any LLM.

This module provides a minimal training abstraction.
Torch is optional - if not available, training functions are stubs.
"""

import sys

# Try to import torch, but don't fail if not available
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class Trainer:
    """Training abstraction for LLM models."""

    def __init__(self, model, opt, tokenizer):
        self.model = model
        self.opt = opt
        self.tok = tokenizer

    def encode(self, t):
        if not TORCH_AVAILABLE:
            return None
        return torch.tensor(self.tok.encode(t).ids)

    def step(self, x, y):
        if not TORCH_AVAILABLE:
            return 0.0
        _, loss = self.model(x, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()


def train(model, trainer, scheduler, steps=200):
    """Train the model for specified steps."""
    if not TORCH_AVAILABLE:
        print("Warning: PyTorch not available, skipping training")
        return

    for i in range(steps):
        stage = scheduler.get_stage(i, steps)
        text = scheduler.sample(stage)
        t = trainer.encode(text)
        if t is None or len(t) < 2:
            continue
        x = t[:-1].unsqueeze(0)
        y = t[1:].unsqueeze(0)
        loss = trainer.step(x, y)
        if i % 25 == 0:
            print(i, stage, loss)


if __name__ == "__main__":
    if TORCH_AVAILABLE:
        print(f"PyTorch available: {torch.__version__}")
    else:
        print("PyTorch not available - training stubs enabled")
