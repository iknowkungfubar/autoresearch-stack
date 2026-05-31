"""Tests for the autonomous loop module."""
import random


class TestAutonomousPipeline:
    """Tests for AutonomousPipeline."""

    def test_init_defaults(self):
        """Test pipeline initializes with default config."""
        from autonomous_loop import AutonomousPipeline

        pipeline = AutonomousPipeline("config.yaml")
        assert pipeline.experiment_count == 0
        assert pipeline.best_val_bpb == float("inf")
        assert pipeline.running is True
        assert pipeline.feedback is not None
        assert pipeline.db is not None
        assert pipeline.memory is not None
        assert pipeline.hypothesis_gen is not None

    def test_prepare_data_basic(self):
        """Test prepare_data with no synthetic generation."""
        from autonomous_loop import AutonomousPipeline

        pipeline = AutonomousPipeline("config.yaml")
        texts = ["hello world test data", "more training data here"]
        data = pipeline.prepare_data(texts, use_synthetic=False, use_model_loop=False)
        assert len(data) >= 1  # At minimum keeps original texts after cleaning

    def test_prepare_data_synthetic(self):
        """Test prepare_data with synthetic generation."""
        from autonomous_loop import AutonomousPipeline

        pipeline = AutonomousPipeline("config.yaml")
        texts = ["hello world test data"]
        data = pipeline.prepare_data(texts, use_synthetic=True, use_model_loop=False)
        # Should have original + synthetic samples
        assert len(data) >= 1

    def test_prepare_curriculum_enabled(self):
        """Test curriculum preparation when enabled."""
        from autonomous_loop import AutonomousPipeline

        pipeline = AutonomousPipeline("config.yaml")
        texts = ["short", "medium length text", "a very long text for testing purposes"]
        scheduler = pipeline.prepare_curriculum(texts)
        assert scheduler is not None

    def test_prepare_curriculum_disabled(self):
        """Test curriculum preparation when disabled."""
        from autonomous_loop import AutonomousPipeline

        # Create pipeline and disable curriculum
        import config as cfg
        cfg.reset_config()
        pipeline = AutonomousPipeline("config.yaml")
        pipeline.config.curriculum.enabled = False
        texts = ["test text"]
        scheduler = pipeline.prepare_curriculum(texts)
        assert scheduler is None

    def test_run_experiment_improves(self):
        """Test run_experiment with random seed for reproducibility."""
        from autonomous_loop import AutonomousPipeline

        pipeline = AutonomousPipeline("config.yaml")
        # Seed for deterministic test
        random.seed(42)
        result = pipeline.run_experiment(
            change_description="Test LR change",
            change_code="config.learning_rate *= 1.1",
            change_type="optimization",
            baseline_val_bpb=1.0,
        )
        assert result["id"] == 1
        assert "val_bpb_before" in result
        assert "val_bpb_after" in result
        assert "status" in result
        assert "improved" in result
        assert pipeline.experiment_count == 1

    def test_run_experiment_tracking(self):
        """Test experiment count increments correctly."""
        from autonomous_loop import AutonomousPipeline

        pipeline = AutonomousPipeline("config.yaml")
        random.seed(123)
        r1 = pipeline.run_experiment("change 1", "code1", "optimization", 1.0)
        r2 = pipeline.run_experiment("change 2", "code2", "optimization", 0.95)
        assert r1["id"] == 1
        assert r2["id"] == 2
        assert pipeline.experiment_count == 2

    def test_get_status(self):
        """Test get_status returns expected fields."""
        from autonomous_loop import AutonomousPipeline

        pipeline = AutonomousPipeline("config.yaml")
        status = pipeline.get_status()
        assert "running" in status
        assert "experiment_count" in status
        assert "best_val_bpb" in status
        assert "statistics" in status
        assert status["running"] is True

    def test_autonomous_pipeline_convenience(self):
        """Test autonomous_pipeline convenience function."""
        from autonomous_loop import autonomous_pipeline

        texts = ["machine learning is a method of data analysis"]
        data = autonomous_pipeline(
            raw=texts,
            config_path="config.yaml",
        )
        assert isinstance(data, list)

    def test_autonomous_pipeline_with_path(self):
        """Test autonomous_pipeline with file path input."""
        import tempfile
        from pathlib import Path
        from autonomous_loop import autonomous_pipeline

        # Create a temp file with sample texts
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("machine learning is fun\nneural networks are powerful\n")
            tmp_path = f.name

        try:
            data = autonomous_pipeline(
                raw=tmp_path,
                config_path="config.yaml",
            )
            assert isinstance(data, list)
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestVectorStore:
    """Tests for vector store (from memory module)."""

    def test_simple_search(self):
        """Test simple vector store search."""
        from memory import SimpleVectorStore, ExperimentMemory

        store = SimpleVectorStore()
        store.add(ExperimentMemory(
            experiment_id=1, timestamp="2026-01-01",
            change_description="learning rate adjustment",
            change_type="optimization", val_bpb_before=1.0,
            val_bpb_after=0.95, status="kept",
        ))
        results = store.search("learning")
        assert len(results) >= 1

    def test_search_empty(self):
        """Test search with empty query returns recent."""
        from memory import SimpleVectorStore, ExperimentMemory

        store = SimpleVectorStore()
        store.add(ExperimentMemory(
            experiment_id=1, timestamp="2026-01-01",
            change_description="test", change_type="test",
            val_bpb_before=1.0, val_bpb_after=0.95, status="kept",
        ))
        results = store.search("")
        assert len(results) >= 1
