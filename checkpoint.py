"""
Checkpoint system for state persistence and resume.

Phase 5: Production Hardening - Checkpoint for resume.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum


class CheckpointStatus(Enum):
    """Checkpoint status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass
class ExperimentCheckpoint:
    """Experiment checkpoint."""

    checkpoint_id: str
    timestamp: str
    experiment_id: int
    iteration: int
    val_bpb: float
    status: str
    state: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass
class SystemCheckpoint:
    """Full system checkpoint."""

    checkpoint_id: str
    timestamp: str
    version: str = "v3.1"
    experiment_checkpoint: Optional[ExperimentCheckpoint] = None
    best_val_bpb: float = float("inf")
    total_experiments: int = 0
    state: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        # Convert checkpoint to dict
        if self.experiment_checkpoint:
            d["experiment_checkpoint"] = asdict(self.experiment_checkpoint)
        return d


class CheckpointManager:
    """Manages checkpoint saving and loading."""

    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.current_checkpoint: Optional[SystemCheckpoint] = None
        self.checkpoints: list = []

        # Load existing checkpoints
        self._load_index()

    def _load_index(self):
        """Load checkpoint index."""
        index_file = self.checkpoint_dir / "index.json"
        if index_file.exists():
            with open(index_file) as f:
                data = json.load(f)
                self.checkpoints = data.get("checkpoints", [])

    def _save_index(self):
        """Save checkpoint index."""
        index_file = self.checkpoint_dir / "index.json"
        with open(index_file, "w") as f:
            json.dump({"checkpoints": self.checkpoints}, f, indent=2)

    def _generate_id(self) -> str:
        """Generate unique checkpoint ID."""
        import uuid

        return f"ckpt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    def create(
        self,
        experiment_id: int,
        iteration: int,
        val_bpb: float,
        status: CheckpointStatus,
        state: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        notes: str = "",
    ) -> str:
        """Create a checkpoint."""
        checkpoint_id = self._generate_id()

        # Create experiment checkpoint
        exp_checkpoint = ExperimentCheckpoint(
            checkpoint_id=checkpoint_id,
            timestamp=datetime.now().isoformat(),
            experiment_id=experiment_id,
            iteration=iteration,
            val_bpb=val_bpb,
            status=status.value,
            state=state or {},
            config=config or {},
            notes=notes,
        )

        # Create system checkpoint
        sys_checkpoint = SystemCheckpoint(
            checkpoint_id=checkpoint_id,
            timestamp=datetime.now().isoformat(),
            experiment_checkpoint=exp_checkpoint,
            best_val_bpb=val_bpb
            if status == CheckpointStatus.COMPLETE
            else float("inf"),
            total_experiments=experiment_id,
        )

        # Save checkpoint
        ckpt_file = self.checkpoint_dir / f"{checkpoint_id}.json"
        with open(ckpt_file, "w") as f:
            json.dump(sys_checkpoint.to_dict(), f, indent=2)

        # Update index
        self.checkpoints.append(
            {
                "id": checkpoint_id,
                "timestamp": sys_checkpoint.timestamp,
                "experiment_id": experiment_id,
                "val_bpb": val_bpb,
                "status": status.value,
            }
        )
        self._save_index()

        self.current_checkpoint = sys_checkpoint

        return checkpoint_id

    def save_progress(
        self,
        experiment_id: int,
        iteration: int,
        val_bpb: float,
        state: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Save progress checkpoint."""
        return self.create(
            experiment_id=experiment_id,
            iteration=iteration,
            val_bpb=val_bpb,
            status=CheckpointStatus.RUNNING,
            state=state,
            config=config,
        )

    def save_complete(
        self,
        experiment_id: int,
        val_bpb: float,
        final_state: Optional[Dict[str, Any]] = None,
    ):
        """Save completion checkpoint."""
        return self.create(
            experiment_id=experiment_id,
            iteration=-1,
            val_bpb=val_bpb,
            status=CheckpointStatus.COMPLETE,
            state=final_state,
        )

    def save_interrupted(
        self,
        experiment_id: int,
        iteration: int,
        val_bpb: float,
        state: Optional[Dict[str, Any]] = None,
    ):
        """Save interrupted checkpoint."""
        return self.create(
            experiment_id=experiment_id,
            iteration=iteration,
            val_bpb=val_bpb,
            status=CheckpointStatus.INTERRUPTED,
            state=state,
            notes="Interrupted by user or system",
        )

    def load(self, checkpoint_id: str) -> Optional[SystemCheckpoint]:
        """Load a checkpoint."""
        ckpt_file = self.checkpoint_dir / f"{checkpoint_id}.json"

        if not ckpt_file.exists():
            return None

        with open(ckpt_file) as f:
            data = json.load(f)

        # Reconstruct
        exp_data = data.get("experiment_checkpoint")
        exp_checkpoint = None
        if exp_data:
            exp_checkpoint = ExperimentCheckpoint(**exp_data)

        return SystemCheckpoint(
            checkpoint_id=data["checkpoint_id"],
            timestamp=data["timestamp"],
            version=data.get("version", "v3.1"),
            experiment_checkpoint=exp_checkpoint,
            best_val_bpb=data.get("best_val_bpb", float("inf")),
            total_experiments=data.get("total_experiments", 0),
            state=data.get("state", {}),
        )

    def load_latest(self) -> Optional[SystemCheckpoint]:
        """Load latest checkpoint."""
        if not self.checkpoints:
            return None

        latest = self.checkpoints[-1]
        return self.load(latest["id"])

    def get_history(self, limit: int = 10) -> list:
        """Get checkpoint history."""
        return self.checkpoints[-limit:]

    def cleanup_old(self, keep: int = 5):
        """Clean up old checkpoints."""
        if len(self.checkpoints) <= keep:
            return

        to_remove = self.checkpoints[:-keep]

        for ckpt in to_remove:
            ckpt_file = self.checkpoint_dir / f"{ckpt['id']}.json"
            if ckpt_file.exists():
                ckpt_file.unlink()

        self.checkpoints = self.checkpoints[-keep:]
        self._save_index()


def get_checkpoint_manager(checkpoint_dir: str = "./checkpoints") -> CheckpointManager:
    """Get checkpoint manager."""
    return CheckpointManager(checkpoint_dir)


if __name__ == "__main__":
    # Test
    print("Testing checkpoint system...")

    mgr = get_checkpoint_manager()

    # Save checkpoint
    ckpt_id = mgr.save_progress(1, 50, 0.95)
    print(f"Created checkpoint: {ckpt_id}")

    # Load checkpoint
    ckpt = mgr.load(ckpt_id)
    print(f"Loaded: val_bpb={ckpt.experiment_checkpoint.val_bpb}")

    # Save completion
    ckpt_id2 = mgr.save_complete(1, 0.92)
    print(f"Complete checkpoint: {ckpt_id2}")

    print("Checkpoint system OK")
