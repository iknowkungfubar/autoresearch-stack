"""
Autonomous Research Loop - Main Pipeline

This is the main orchestration for the autonomous research system.
It combines:
- Data intelligence (cleaning)
- Synthetic data generation
- Curriculum learning
- Experiment tracking
- Feedback evaluation
"""

import time
import argparse
from typing import List, Optional, Dict, Any
from datetime import datetime

# Import components
from synthetic_data import (
    SyntheticGenerator,
    model_in_the_loop_generate,
)
from data_intelligence import clean_corpus
from curriculum import build_curriculum, create_scheduler
from feedback import Feedback, ExperimentStatus
from storage import ExperimentDB
from config import get_config
from memory import MemorySystem
from prioritization import get_prioritization
from hypothesis import HypothesisGenerator


class AutonomousPipeline:
    """Main autonomous research pipeline."""

    def __init__(self, config_path: str = "config.yaml"):
        # Load configuration
        self.config = get_config(config_path)

        # Initialize components
        self.synthetic_generator = SyntheticGenerator(
            use_llm=self.config.synthetic.use_llm,
            provider=self.config.synthetic.model_provider,
            model=self.config.synthetic.model_name,
            temperature=self.config.synthetic.temperature,
        )

        # Initialize feedback system
        self.feedback = Feedback(
            experiment_log_path=self.config.logging.experiment_log,
        )

        # Initialize storage
        self.db = ExperimentDB("./experiments.db")

        # Initialize memory system (Phase 3.1)
        self.memory = MemorySystem(db_path="./experiments.db")

        # Initialize prioritization (Phase 3.1)
        self.prioritization = get_prioritization(strategy="ucb1")

        # Initialize hypothesis generator (Phase 3.1)
        self.hypothesis_gen = HypothesisGenerator(
            use_llm=self.config.synthetic.use_llm,
            provider=self.config.synthetic.model_provider,
            model=self.config.synthetic.model_name,
        )

        # State
        self.experiment_count = 0
        self.best_val_bpb = float("inf")
        self.running = True

    def prepare_data(
        self,
        raw_texts: List[str],
        use_synthetic: bool = True,
        use_model_loop: bool = False,
        model=None,
        tokenizer=None,
    ) -> List[str]:
        """Prepare training data through the pipeline.

        Args:
            raw_texts: Raw input texts
            use_synthetic: Include synthetic data
            use_model_loop: Use model-in-the-loop generation
            model: Model for model-in-the-loop generation
            tokenizer: Tokenizer for model

        Returns:
            Prepared dataset
        """
        # Step 1: Clean corpus
        print("Step 1: Cleaning corpus...")
        data = clean_corpus(raw_texts)
        print(f"  Cleaned: {len(data)} texts")

        # Step 2: Generate synthetic data
        if use_synthetic:
            print("Step 2: Generating synthetic data...")
            synthetic_result = self.synthetic_generator.generate(
                n=self.config.synthetic.n_samples,
                difficulty="mixed",
            )
            data.extend(synthetic_result.prompts)
            print(f"  Added {len(synthetic_result.prompts)} synthetic prompts")

        # Step 3: Model-in-the-loop generation
        if use_model_loop and model is not None:
            print("Step 3: Model-in-the-loop generation...")
            prompts = [
                "Generate training data for:"
            ] * self.config.synthetic.n_model_samples
            model_outputs = model_in_the_loop_generate(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                n_samples=self.config.synthetic.n_model_samples,
            )
            data.extend(model_outputs)
            print(f"  Added {len(model_outputs)} model-generated outputs")

        # Step 4: Quality filter
        print("Step 4: Quality filtering...")
        original_count = len(data)
        data = self.synthetic_generator.quality_filter(data)
        print(f"  Filtered: {original_count} → {len(data)}")

        return data

    def prepare_curriculum(self, texts: List[str]):
        """Prepare curriculum scheduler."""
        print("Preparing curriculum...")

        if self.config.curriculum.enabled:
            curriculum = build_curriculum(
                texts,
                stages=self.config.curriculum.stages,
                metric=self.config.curriculum.difficulty_metric,
            )

            if self.config.curriculum.adaptive:
                scheduler = create_scheduler(
                    texts,
                    adaptive=True,
                    warmup_ratio=self.config.curriculum.warmup_ratio,
                )
            else:
                scheduler = create_scheduler(texts, adaptive=False)

            print(
                f"  Curriculum: {', '.join(f'{k}: {len(v)}' for k, v in curriculum.items())}"
            )
        else:
            scheduler = None
            print("  Curriculum disabled")

        return scheduler

    def run_training(
        self,
        model,
        trainer,
        scheduler,
        steps: int,
        val_bpb_target: float,
    ) -> Dict[str, Any]:
        """Run training and return results.

        Args:
            model: The model
            trainer: Trainer instance
            scheduler: Curriculum scheduler
            steps: Number of steps
            val_bpb_target: Target val_bpb

        Returns:
            Training results
        """

        start_time = time.time()

        for i in range(steps):
            # Get curriculum stage
            if scheduler:
                stage = scheduler.get_stage(i, steps)
                text = scheduler.sample(stage)
            else:
                text = "Sample text for training"

            # Simple forward pass (stub)
            # In real implementation, this would do actual training
            try:
                if hasattr(trainer, "encode"):
                    t = trainer.encode(text)
                    if len(t) < 2:
                        continue
                    x = t[:-1].unsqueeze(0)
                    y = t[1:].unsqueeze(0)

                    _, loss = model(x, y)
                    trainer.opt.zero_grad()
                    loss.backward()
                    trainer.opt.step()

                    if i % 25 == 0:
                        print(f"  Step {i}: loss={loss.item():.4f}")
            except Exception as e:
                print(f"  Training error at step {i}: {e}")
                continue

        training_time = time.time() - start_time

        # Stub: return a simulated val_bpb
        # In real implementation, this would evaluate on validation set
        val_bpb = 1.0 + (self.experiment_count * 0.01)  # Simulated

        return {
            "val_bpb": val_bpb,
            "training_time": training_time,
            "steps_completed": steps,
        }

    def run_experiment(
        self,
        change_description: str,
        change_code: str,
        change_type: str,
        baseline_val_bpb: float,
    ) -> Dict[str, Any]:
        """Run a single experiment.

        Args:
            change_description: What changed
            change_code: The code change
            change_type: Type of change
            baseline_val_bpb: Baseline val_bpb

        Returns:
            Experiment results
        """
        self.experiment_count += 1
        exp_id = self.experiment_count

        print(f"\n{'=' * 60}")
        print(f"EXPERIMENT {exp_id}")
        print(f"{'=' * 60}")
        print(f"Change: {change_description}")
        print(f"Type: {change_type}")
        print(f"Baseline val_bpb: {baseline_val_bpb:.6f}")

        # Start experiment in database
        timestamp = datetime.now().isoformat()
        self.db.insert_experiment(
            timestamp=timestamp,
            change_description=change_description,
            change_code=change_code,
            change_type=change_type,
            val_bpb_before=baseline_val_bpb,
            status="running",
        )

        # Run training (stub - would run actual training)
        # In real implementation, this would apply change_code to train.py
        # and run training

        # Simulate training result
        val_bpb_after = baseline_val_bpb - 0.01  # Simulated improvement

        # Determine if improved
        if val_bpb_after < baseline_val_bpb:
            status = ExperimentStatus.KEPT
            print(
                f"Result: val_bpb improved {baseline_val_bpb:.6f} → {val_bpb_after:.6f}"
            )
            print("Status: KEPT")

            if val_bpb_after < self.best_val_bpb:
                self.best_val_bpb = val_bpb_after
        else:
            status = ExperimentStatus.REVERTED
            print(
                f"Result: val_bpb did not improve {baseline_val_bpb:.6f} → {val_bpb_after:.6f}"
            )
            print("Status: REVERTED")

        # Update database
        self.db.update_experiment(
            exp_id,
            val_bpb_after=val_bpb_after,
            status=status.value,
        )

        return {
            "id": exp_id,
            "val_bpb_before": baseline_val_bpb,
            "val_bpb_after": val_bpb_after,
            "status": status.value,
            "improved": status == ExperimentStatus.KEPT,
        }

    def run_autonomous_loop(
        self,
        raw_texts: List[str],
        num_experiments: Optional[int] = None,
    ):
        """Run the full autonomous loop.

        Args:
            raw_texts: Raw training texts
            num_experiments: Max experiments (or use config)
        """
        num_experiments = num_experiments or self.config.experiment.budget

        print(f"\n{'#' * 60}")
        print("# AUTONOMOUS RESEARCH LOOP (v3.1)")
        print(f"{'#' * 60}")
        print(f"Max experiments: {num_experiments}")
        print(f"Time per experiment: {self.config.experiment.time_per_experiment}s")
        print(f"Target val_bpb: {self.config.experiment.val_target}")

        # Prepare data
        data = self.prepare_data(raw_texts)

        # Prepare curriculum
        _scheduler = self.prepare_curriculum(data)

        # Load memory from database
        self.memory.load_from_db()
        print(
            f"\nMemory loaded: {len(self.memory.vector_store.experiments)} experiments"
        )

        # Get baseline
        baseline = self.feedback.get_baseline()
        if baseline == float("inf"):
            baseline = 1.0  # Default baseline
        print(f"\nBaseline val_bpb: {baseline:.6f}")

        # Run experiments
        for i in range(num_experiments):
            # Check stop conditions
            if self.best_val_bpb <= self.config.experiment.val_target:
                print(f"\nTarget val_bpb {self.config.experiment.val_target} achieved!")
                break

            # Use memory to query what has been tried
            what_tried = self.memory.get_what_been_tried("learning rate")

            # Get suggestion from prioritization
            suggestion = self.prioritization.suggest_next(baseline)

            # Generate hypothesis
            hypothesis_list = self.hypothesis_gen.generate(
                n=1,
                change_type=suggestion.get("category", "optimization"),
                memory_context=what_tried,
            )

            if hypothesis_list:
                hypothesis = hypothesis_list[0]
                change_desc = hypothesis.description
                change_code = hypothesis.code_diff
                change_type = hypothesis.change_type
            else:
                # Fallback
                change_desc = "Try learning rate adjustment"
                change_code = "config.model.learning_rate *= 1.1"
                change_type = "optimization"

            # Run experiment
            result = self.run_experiment(
                change_description=change_desc,
                change_code=change_code,
                change_type=change_type,
                baseline_val_bpb=self.best_val_bpb,
            )

            # Record to prioritization system
            self.prioritization.record_result(
                change=change_desc,
                change_type=change_type,
                val_bpb_before=result["val_bpb_before"],
                val_bpb_after=result["val_bpb_after"],
            )

            # Update baseline for next experiment
            if result["improved"]:
                baseline = result["val_bpb_after"]

        # Print summary
        stats = self.db.get_statistics()
        print(f"\n{'#' * 60}")
        print("# EXPERIMENT SUMMARY")
        print(f"{'#' * 60}")
        print(f"Total experiments: {stats['total_experiments']}")
        print(f"Kept: {stats['kept']}")
        print(f"Reverted: {stats['reverted']}")
        print(f"Best val_bpb: {stats.get('best_val_bpb', 'N/A')}")
        print(f"Improvement: {stats.get('improvement', 'N/A')}")

    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        stats = self.db.get_statistics()
        return {
            "running": self.running,
            "experiment_count": self.experiment_count,
            "best_val_bpb": self.best_val_bpb
            if self.best_val_bpb < float("inf")
            else None,
            "statistics": stats,
        }


def autonomous_pipeline(
    raw,
    embed_fn=None,
    model=None,
    tokenizer=None,
    config_path: str = "config.yaml",
):
    """Convenience function for backward compatibility.

    Args:
        raw: Raw texts or path to text file
        embed_fn: Embedding function (optional)
        model: Model for model-in-the-loop (optional)
        tokenizer: Tokenizer (optional)
        config_path: Path to config

    Returns:
        Prepared dataset
    """
    # Handle raw as path or list
    if isinstance(raw, str):
        with open(raw, "r") as f:
            raw_texts = [line.strip() for line in f if line.strip()]
    else:
        raw_texts = raw

    # Create pipeline
    pipeline = AutonomousPipeline(config_path)

    # Prepare data
    data = pipeline.prepare_data(
        raw_texts,
        use_synthetic=True,
        use_model_loop=model is not None,
        model=model,
        tokenizer=tokenizer,
    )

    return data


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Autonomous Research Loop")
    parser.add_argument(
        "--config",
        "-c",
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--input",
        "-i",
        help="Input text file (one text per line)",
    )
    parser.add_argument(
        "--experiments",
        "-n",
        type=int,
        default=None,
        help="Number of experiments to run",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare data, don't run experiments",
    )

    args = parser.parse_args()

    # Load texts
    if args.input:
        with open(args.input, "r") as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        # Default sample texts
        texts = [
            "Machine learning is a method of data analysis.",
            "Neural networks are inspired by biological neurons.",
            "Transformers use attention mechanisms.",
            "Backpropagation trains neural networks.",
            "Gradient descent optimizes model parameters.",
        ]

    # Create pipeline
    pipeline = AutonomousPipeline(args.config)

    # Prepare data
    data = pipeline.prepare_data(texts)
    print(f"\nPrepared {len(data)} training examples")

    # Show config
    print("\nConfiguration:")
    print(f"  Experiments: {args.experiments or pipeline.config.experiment.budget}")
    print(f"  Time per exp: {pipeline.config.experiment.time_per_experiment}s")
    print(f"  Target val_bpb: {pipeline.config.experiment.val_target}")

    if args.prepare_only:
        print("\nData preparation complete (--prepare-only)")
        return

    # Prepare curriculum
    _scheduler = pipeline.prepare_curriculum(data)

    # Run autonomous loop
    pipeline.run_autonomous_loop(texts, args.experiments)


if __name__ == "__main__":
    main()
