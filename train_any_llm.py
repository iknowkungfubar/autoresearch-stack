"""
Training abstraction for any LLM.

This module provides a training abstraction that works with or without PyTorch.
- Without PyTorch: uses a numpy-based demo model (linear regression) to
  demonstrate the training pipeline API and validate experiment logic.
- With PyTorch: trains actual LLM models when torch is installed.

The demo mode is designed so every component of the autonomous research stack
can be exercised: curriculum, feedback, checkpointing, memory, etc.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List

# Try to import numpy (required for demo mode)
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore[assignment]

# Try to import torch as optional dependency
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]


@dataclass
class TrainingResult:
    """Result of a training run."""

    val_bpb: float
    training_loss: float
    eval_loss: float
    steps_completed: int
    training_time: float
    converged: bool


# ── Numpy Demo Model ──────────────────────────────────────────


class NumpyDemoModel:
    """Simple numpy-based demo model for testing the training pipeline.

    Uses linear regression on synthetic 2D data. The model learns
    from curriculum-provided text by encoding text length as a
    regression target. This lets all pipeline components (curriculum,
    feedback, hypothesis, memory) receive realistic training signals
    without requiring PyTorch.

    Public API matches what AutonomousPipeline.run_training() expects:
    - model(x, y) -> (predictions, loss)
    """

    def __init__(self, learning_rate: float = 0.01):
        import numpy as np

        self.rng = np.random.default_rng(42)
        # Simple linear parameters
        self.w = self.rng.standard_normal(1) * 0.1
        self.b = np.float64(0.0)
        self.lr = learning_rate

    def forward(self, x):
        """Forward pass: y = w * x + b (scaled for demo)."""
        return self.w * x + self.b

    def __call__(self, x, y):
        """Compute forward pass and MSE loss.

        Args:
            x: Input tensor/array
            y: Target tensor/array

        Returns:
            Tuple of (predictions, loss_value)
        """
        import numpy as np

        x_np = np.asarray(x, dtype=np.float64)
        y_np = np.asarray(y, dtype=np.float64)

        preds = self.forward(x_np)
        loss = np.mean((preds - y_np) ** 2)

        # Gradient step (simple SGD)
        grad_w = np.mean(2 * (preds - y_np) * x_np)
        grad_b = np.mean(2 * (preds - y_np))
        self.w -= self.lr * grad_w
        self.b -= self.lr * grad_b

        return preds, loss

    def state_dict(self) -> Dict[str, Any]:
        """Return model state for checkpointing."""
        return {
            "w": float(self.w.item()) if hasattr(self.w, "item") else float(self.w),
            "b": float(self.b.item()) if hasattr(self.b, "item") else float(self.b),
            "lr": self.lr,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """Load model state from checkpoint."""
        self.w = state.get("w", self.w)
        self.b = state.get("b", self.b)
        self.lr = state.get("lr", self.lr)


# ── Trainer ───────────────────────────────────────────────────


class Trainer:
    """Training abstraction for LLM models (or numpy demo).

    Args:
        model: Either a PyTorch model or NumpyDemoModel
        opt: PyTorch optimizer (required for torch models)
        tokenizer: Tokenizer (required for torch models)
    """

    def __init__(self, model, opt=None, tokenizer=None):
        self.model = model
        self.opt = opt
        self.tok = tokenizer

    def encode(self, text: str):
        """Encode text for training.

        For torch models: uses the tokenizer to produce token IDs.
        For numpy demo: encodes text length as a feature (for regression).
        """
        if TORCH_AVAILABLE and self.tok is not None:
            return torch.tensor(self.tok.encode(text).ids)  # type: ignore[union-attr]
        else:
            # Numpy demo: encode text as length-based features
            import numpy as np

            length = len(text)
            # Return a synthetic 1D input derived from text
            return np.array([length / 100.0], dtype=np.float64)

    def step(self, x, y) -> float:
        """Run one training step.

        Args:
            x: Input data
            y: Target data

        Returns:
            Loss value (scalar)
        """
        if TORCH_AVAILABLE and self.opt is not None:
            _, loss = self.model(x, y)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            return float(loss.item())
        else:
            _, loss = self.model(x, y)
            return float(loss)

    def get_lr(self) -> float:
        """Get current learning rate."""
        if hasattr(self.model, "lr"):
            return self.model.lr
        if self.opt is not None and hasattr(self.opt, "param_groups"):
            return self.opt.param_groups[0]["lr"]
        return 0.01


def train(
    model,
    trainer: Trainer,
    scheduler,
    steps: int = 200,
    eval_every: int = 25,
    val_bpb_target: float = 0.95,
) -> TrainingResult:
    """Train the model for specified steps.

    Args:
        model: Model instance (torch or numpy demo).
        trainer: Trainer wrapping the model.
        scheduler: Curriculum scheduler providing text samples.
        steps: Number of training steps.
        eval_every: Compute eval metrics every N steps.
        val_bpb_target: Target validation loss for convergence.

    Returns:
        TrainingResult with final metrics.
    """
    start_time = time.time()
    losses: List[float] = []

    for i in range(steps):
        # Get curriculum stage and sample text
        stage = scheduler.get_stage(i, steps)
        text = scheduler.sample(stage)

        # Encode and create input/target pair
        t = trainer.encode(text)
        if t is None:
            continue

        # For numpy demo: target is the input with small noise
        # (learning objective: predict input features)
        import numpy as np

        if isinstance(t, np.ndarray):
            x = t.reshape(1, -1)
            y = t + np.random.default_rng().normal(0, 0.01, t.shape)
            y = y.reshape(1, -1)
        else:
            # PyTorch path
            if len(t) < 2:
                continue
            x = t[:-1].unsqueeze(0)
            y = t[1:].unsqueeze(0)

        loss = trainer.step(x, y)
        losses.append(loss)

        if i % eval_every == 0:
            print(f"  Step {i}: loss={loss:.6f}, stage={stage}")

    training_time = time.time() - start_time

    # Compute training metrics
    avg_loss = float(np.mean(losses[-10:])) if losses else 1.0
    final_loss = float(losses[-1]) if losses else 1.0

    # Simulate eval on a held-out metric
    # In real training, this would be val_bpb from the model
    # For the numpy demo, derive from training dynamics
    improvement = max(0, 1.0 - final_loss)
    val_bpb = max(0.5, 1.5 - improvement * 0.8)

    converged = val_bpb <= val_bpb_target or final_loss < 0.05

    return TrainingResult(
        val_bpb=val_bpb,
        training_loss=avg_loss,
        eval_loss=final_loss * 1.1,  # Slightly higher than training (simulated)
        steps_completed=steps,
        training_time=training_time,
        converged=converged,
    )


def create_demo_model() -> NumpyDemoModel:
    """Create a numpy demo model for testing without PyTorch."""
    return NumpyDemoModel(learning_rate=0.01)


def demo_training():
    """Run a demo training session using the numpy model.

    This demonstrates the training API and can be used to
    validate the pipeline without installing PyTorch.
    """
    from curriculum import Scheduler

    print("=" * 50)
    print("NUMPY DEMO TRAINING")
    print("=" * 50)

    # Create demo model and trainer
    model = create_demo_model()
    trainer = Trainer(model=model)

    # Create a simple curriculum
    curriculum = {
        "easy": ["a" * 20, "b" * 25, "c" * 30],
        "medium": ["d" * 50, "e" * 60, "f" * 70],
        "hard": ["g" * 100, "h" * 120, "i" * 150],
    }
    scheduler = Scheduler(curriculum)

    # Train
    result = train(model, trainer, scheduler, steps=100)

    print("\nTraining complete:")
    print(f"  Steps: {result.steps_completed}")
    print(f"  Final training loss: {result.training_loss:.6f}")
    print(f"  Eval loss: {result.eval_loss:.6f}")
    print(f"  Val BPB: {result.val_bpb:.4f}")
    print(f"  Training time: {result.training_time:.2f}s")
    print(f"  Converged: {result.converged}")

    return result


if __name__ == "__main__":
    import sys

    if "--demo" in sys.argv:
        demo_training()
    elif TORCH_AVAILABLE:
        print(f"PyTorch available: {torch.__version__}")
        print("Run with --demo for numpy demo training")
    else:
        print("PyTorch not available — running numpy demo mode")
        demo_training()
