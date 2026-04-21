"""
Test suite for Autonomous Research Stack.

Unit tests for core modules.
"""

import pytest
import os


# Test config
class TestConfig:
    """Test configuration loading."""

    def test_config_load(self):
        """Test config loads from yaml."""
        from config import get_config

        c = get_config("config.yaml")
        assert c.experiment.budget == 500
        assert c.experiment.time_per_experiment == 300

    def test_config_env_override(self):
        """Test environment variable override."""
        os.environ["EXPERIMENT_BUDGET"] = "100"
        from config import reset_config, get_config

        reset_config()
        c = get_config("config.yaml")
        assert c.experiment.budget == 100
        del os.environ["EXPERIMENT_BUDGET"]

    def test_config_to_dict(self):
        """Test config serialization."""
        from config import get_config

        c = get_config("config.yaml")
        d = c.to_dict()
        assert "experiment" in d
        assert "model" in d


# Test data_intelligence
class TestDataIntelligence:
    """Test data intelligence module."""

    def test_repair(self):
        """Test text repair."""
        from data_intelligence import repair

        # Note: repair requires length >= 20
        assert repair("a" * 20) is not None
        assert repair("") is None
        assert repair("short") is None  # too short
        assert repair("hello\x00world with more text") is not None

    def test_is_noise(self):
        """Test noise detection."""
        from data_intelligence import is_noise

        assert is_noise("aaa") is True  # too few unique
        assert is_noise("123!@#") is True  # no alpha
        assert is_noise("hello world test") is False

    def test_clean_corpus(self):
        """Test corpus cleaning."""
        from data_intelligence import clean_corpus

        texts = [
            "hello world test data here",
            "",
            "aaa",
            "valid text content for testing",
        ]
        result = clean_corpus(texts)
        # Some may pass, some may fail based on noise detection
        assert isinstance(result, list)


# Test synthetic_data
class TestSyntheticData:
    """Test synthetic data generation."""

    def test_generate_synthetic(self):
        """Test synthetic generation."""
        from synthetic_data import generate_synthetic

        result = generate_synthetic(n=10, difficulty="easy")
        assert len(result) == 10

    def test_synthetic_generator(self):
        """Test SyntheticGenerator."""
        from synthetic_data import SyntheticGenerator

        gen = SyntheticGenerator(use_llm=False)
        result = gen.generate(n=5)
        assert len(result.prompts) == 5
        assert result.used_llm is False

    def test_quality_filter(self):
        """Test quality filtering."""
        from synthetic_data import SyntheticGenerator

        gen = SyntheticGenerator()
        prompts = ["short", "a" * 100, "", "valid prompt text"]
        filtered = gen.quality_filter(prompts, min_length=10, max_length=50)
        assert len(filtered) <= len(prompts)


# Test curriculum
class TestCurriculum:
    """Test curriculum module."""

    def test_compute_difficulty(self):
        """Test difficulty computation."""
        from curriculum import compute_difficulty

        assert compute_difficulty("short") > 0
        assert compute_difficulty("much longer text here") > compute_difficulty("short")

    def test_build_curriculum(self):
        """Test curriculum building."""
        from curriculum import build_curriculum

        texts = ["a", "bb", "ccc", "dddd", "eeeee", "ffffff"]
        curriculum = build_curriculum(texts, stages=3)
        assert "easy" in curriculum
        assert "medium" in curriculum
        assert "hard" in curriculum

    def test_scheduler(self):
        """Test Scheduler."""
        from curriculum import Scheduler

        curriculum = {"easy": ["a"], "medium": ["b"], "hard": ["c"]}
        scheduler = Scheduler(curriculum)
        assert scheduler.get_stage(0, 100) == "easy"
        assert scheduler.get_stage(50, 100) == "medium"
        assert scheduler.get_stage(90, 100) == "hard"


# Test feedback
class TestFeedback:
    """Test feedback module."""

    def test_reward(self):
        """Test reward computation."""
        from feedback import Feedback

        fb = Feedback()
        reward = fb.reward(1.0, 0.0)
        assert reward > 0

    def test_classify_failure(self):
        """Test failure classification."""
        from feedback import Feedback, FailureClassification

        fb = Feedback()
        classification = fb.classify_failure(
            val_bpb_before=1.0,
            val_bpb_after=1.5,
            training_loss=0.1,
            eval_loss=5.0,
            training_stable=True,
        )
        assert classification == FailureClassification.OVERFITTING


# Test storage
class TestStorage:
    """Test storage module."""

    def test_experiment_db(self, tmp_path):
        """Test SQLite storage."""
        from storage import ExperimentDB

        db_path = tmp_path / "test.db"
        db = ExperimentDB(str(db_path))

        # Insert experiment
        exp_id = db.insert_experiment(
            timestamp="2026-01-01",
            change_description="Test change",
            change_code="code",
            change_type="optimization",
            val_bpb_before=1.0,
            status="running",
        )
        assert exp_id > 0

        # Update experiment
        db.update_experiment(exp_id, val_bpb_after=0.95, status="kept")

        # Get experiment
        exp = db.get_experiment(exp_id)
        assert exp is not None
        assert exp["val_bpb_after"] == 0.95


# Test memory
class TestMemory:
    """Test memory module."""

    def test_simple_vector_store(self):
        """Test vector store."""
        from memory import SimpleVectorStore, ExperimentMemory

        store = SimpleVectorStore()

        exp = ExperimentMemory(
            experiment_id=1,
            timestamp="2026-01-01",
            change_description="Test learning rate",
            change_type="optimization",
            val_bpb_before=1.0,
            val_bpb_after=0.95,
            status="kept",
        )

        store.add(exp)

        # Search
        results = store.search("learning")
        assert len(results) >= 1


# Test prioritization
class TestPrioritization:
    """Test prioritization module."""

    def test_bandit_selector(self):
        """Test bandit selection."""
        from prioritization import BanditSelector

        selector = BanditSelector(strategy="ucb1")

        # Initial selection
        arm = selector.select("optimization")
        assert arm in [
            "learning_rate",
            "batch_size",
            "weight_decay",
            "warmup_steps",
            "optimizer",
        ]

        # Update
        selector.update("learning_rate", 1.0, "optimization")

        stats = selector.get_statistics("optimization")
        assert "learning_rate" in stats


# Test hypothesis
class TestHypothesis:
    """Test hypothesis generation."""

    def test_generate_hypothesis(self):
        """Test hypothesis generation."""
        from hypothesis import HypothesisGenerator

        gen = HypothesisGenerator(use_llm=False)
        hypotheses = gen.generate(n=3, change_type="optimization")

        assert len(hypotheses) == 3
        for h in hypotheses:
            assert h.change_type == "optimization"

    def test_analysis_hypothesis(self):
        """Test analysis-based hypothesis."""
        from hypothesis import HypothesisGenerator

        gen = HypothesisGenerator()
        h = gen.generate_from_analysis(6.0, 2.0)  # High loss, high val_bpb

        assert h is not None


# Test sandbox
class TestSandbox:
    """Test sandbox module."""

    def test_safe_runner(self):
        """Test safe runner."""
        from sandbox import SafeRunner

        runner = SafeRunner()

        # Test safe code
        result = runner.run("print('hello')")
        assert result.success is True
        assert "hello" in result.stdout

        # Test blocked code
        result = runner.run("import os")
        assert result.success is False


# Test checkpoint
class TestCheckpoint:
    """Test checkpoint module."""

    def test_checkpoint_manager(self, tmp_path):
        """Test checkpoint saving."""
        from checkpoint import CheckpointManager

        mgr = CheckpointManager(str(tmp_path / "checkpoints"))

        # Save progress
        ckpt_id = mgr.save_progress(
            experiment_id=1,
            iteration=50,
            val_bpb=0.95,
        )

        assert ckpt_id is not None

        # Load
        ckpt = mgr.load(ckpt_id)
        assert ckpt is not None
        assert ckpt.experiment_checkpoint.val_bpb == 0.95


# Test monitor
class TestMonitor:
    """Test monitor module."""

    def test_monitor(self):
        """Test monitor."""
        from monitor import Monitor

        m = Monitor()

        # Start experiment
        m.start_experiment(1, "Test", 1.0)
        assert m.stats.running == 1

        # Complete
        m.complete_experiment(0.95, "kept")
        assert m.stats.kept == 1
        assert m.stats.total_experiments == 1


# Test report
class TestReport:
    """Test report module."""

    def test_report_generation(self):
        """Test report generation."""
        from report import Report

        r = Report("Test Report")
        r.add_section("Overview", "Test content")

        content = r.render()
        assert "Test Report" in content
        assert "Overview" in content

    def test_summary_report(self):
        """Test summary report."""
        from report import generate_summary_report

        experiments = [
            {
                "id": 1,
                "change_description": "Test",
                "change_type": "opt",
                "status": "kept",
                "val_bpb_after": 0.95,
            },
        ]

        report = generate_summary_report(experiments)
        content = report.render()
        assert "kept" in content.lower()


# Test multi_agent
class TestMultiAgent:
    """Test multi-agent module."""

    def test_orchestrator(self):
        """Test orchestrator."""
        from multi_agent import OrchestratorAgent

        orch = OrchestratorAgent()

        # Run a cycle
        result = orch.run_experiment_cycle({"val_bpb": 1.0})

        assert result is not None
        assert "result" in result
        assert "statistics" in result
        assert result["result"]["improved"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
