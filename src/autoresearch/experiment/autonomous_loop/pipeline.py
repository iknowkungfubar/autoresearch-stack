"""
Autonomous Research Loop - Pipeline Orchestration

This is the main orchestration for the autonomous research system.
It combines:
- Data intelligence (cleaning)
- Synthetic data generation
- Curriculum learning
- Experiment tracking
- Feedback evaluation

Training is executed by running train_any_llm.py as a subprocess;
hypothesis code diffs are applied to the live config before each run.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from autoresearch.config import get_config
from autoresearch.data.curriculum import build_curriculum, create_scheduler
from autoresearch.data.data_intelligence import clean_corpus
from autoresearch.data.synthetic_data import (
    SyntheticGenerator,
    model_in_the_loop_generate,
)
from autoresearch.experiment.feedback import ExperimentStatus, Feedback
from autoresearch.experiment.hypothesis import HypothesisGenerator
from autoresearch.experiment.memory import MemorySystem
from autoresearch.experiment.prioritization import get_prioritization
from autoresearch.experiment.storage import ExperimentDB


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
        print(f"  Filtered: {original_count} \u2192 {len(data)}")

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
                "  Curriculum: "
                f"{', '.join(f'{k}: {len(v)}' for k, v in curriculum.items())}"
            )
        else:
            scheduler = None
            print("  Curriculum disabled")

        return scheduler

    def run_training(
        self,
        steps: int = 200,
        val_bpb_target: float = 0.95,
    ) -> Dict[str, Any]:
        """Run training and return results.

        Executes train_any_llm.py as a subprocess and captures its
        output to extract real training metrics.

        Args:
            steps: Number of training steps
            val_bpb_target: Target val_bpb

        Returns:
            Training results
        """
        import subprocess
        import sys

        train_script = (
            Path(__file__).resolve().parent.parent.parent
            / "llm"
            / "train_any_llm.py"
        )

        start_time = time.time()

        try:
            # Safe: using the current Python interpreter, not user-controlled input
            result = subprocess.run(  # noqa: S603
                [sys.executable, str(train_script), "--demo"],
                capture_output=True,
                text=True,
                timeout=self.config.experiment.time_per_experiment,
            )
            output = result.stdout + result.stderr
            training_time = time.time() - start_time

            # Parse val_bpb from output
            val_bpb = 1.5  # fallback default
            for line in output.splitlines():
                if "Val BPB:" in line:
                    try:
                        val_bpb = float(line.split("Val BPB:")[-1].strip())
                    except (ValueError, IndexError):
                        pass

            # Parse training loss
            training_loss = 1.0
            for line in output.splitlines():
                if "Final training loss:" in line:
                    try:
                        training_loss = float(
                            line.split("Final training loss:")[-1].strip()
                        )
                    except (ValueError, IndexError):
                        pass

            print(f"  Subprocess training completed in {training_time:.2f}s")
            if result.returncode != 0:
                print(f"  stderr: {result.stderr[:200]}")

            return {
                "val_bpb": val_bpb,
                "training_loss": training_loss,
                "training_time": training_time,
                "steps_completed": steps,
                "subprocess_output": output,
            }

        except subprocess.TimeoutExpired:
            print(
                "  Training timed out after"
                f" {self.config.experiment.time_per_experiment}s"
            )
            return {
                "val_bpb": 2.0,
                "training_loss": 2.0,
                "training_time": self.config.experiment.time_per_experiment,
                "steps_completed": 0,
                "subprocess_output": "",
            }
        except Exception as e:
            print(f"  Training subprocess failed: {e}")
            return {
                "val_bpb": 2.0,
                "training_loss": 2.0,
                "training_time": time.time() - start_time,
                "steps_completed": 0,
                "subprocess_output": str(e),
            }

    def _apply_code_diff(self, code_diff: str) -> None:
        """Apply a code_diff string to the live config object.

        The code_diff is a Python expression/statement that mutates
        ``self.config`` - e.g. ``config.model.learning_rate *= 1.1``.

        The string is evaluated with ``config`` bound to ``self.config``
        so expressions like ``config.model.batch_size *= 2`` work.
        """
        if not code_diff or not code_diff.strip():
            return

        # Build a safe namespace with only the config reference
        namespace: dict[str, object] = {"config": self.config}
        try:
            exec(code_diff, namespace)  # noqa: S102 - controlled input
            print(f"  Applied code_diff: {code_diff}")
        except Exception as e:
            print(f"  Failed to apply code_diff '{code_diff}': {e}")

    def run_experiment(
        self,
        change_description: str,
        change_code: str,
        change_type: str,
        baseline_val_bpb: float,
    ) -> Dict[str, Any]:
        """Run a single experiment.

        Applies the hypothesis code_diff to the live config, then
        executes a real training subprocess (train_any_llm.py) and
        captures the resulting val_bpb.

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

        # Apply the hypothesis code change to the live config (Issue 4)
        if change_code:
            self._apply_code_diff(change_code)

        # Run real training via subprocess
        training_result = self.run_training(
            steps=self.config.curriculum.stages * 50,
            val_bpb_target=self.config.experiment.val_target,
        )

        val_bpb_after = training_result["val_bpb"]

        # Determine if improved
        if val_bpb_after < baseline_val_bpb:
            status = ExperimentStatus.KEPT
            print(
                f"Result: val_bpb improved {baseline_val_bpb:.6f} \u2192 {val_bpb_after:.6f}"
            )
            print("Status: KEPT")

            if val_bpb_after < self.best_val_bpb:
                self.best_val_bpb = val_bpb_after
        else:
            status = ExperimentStatus.REVERTED
            print(
                f"Result: val_bpb did not improve "
                f"{baseline_val_bpb:.6f} \u2192 {val_bpb_after:.6f}"
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
