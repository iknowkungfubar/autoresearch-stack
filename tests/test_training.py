"""Tests for the training module (train_any_llm.py).

Covers:
- NumpyDemoModel forward/backward/state persistence
- Trainer encode/step/get_lr for both numpy and torch paths
- train() function with curriculum scheduler
- demo_training() end-to-end
- TrainingResult dataclass
"""
import numpy as np


class TestNumpyDemoModel:
    """Tests for the numpy-based demo model."""

    def test_init(self):
        from train_any_llm import NumpyDemoModel

        model = NumpyDemoModel(learning_rate=0.01)
        assert model.lr == 0.01
        assert model.w is not None
        assert model.b is not None

    def test_forward(self):
        from train_any_llm import NumpyDemoModel

        model = NumpyDemoModel()
        x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        preds = model.forward(x)
        assert preds.shape == x.shape
        assert np.all(np.isfinite(preds))

    def test_call(self):
        from train_any_llm import NumpyDemoModel

        model = NumpyDemoModel(learning_rate=0.1)
        x = np.array([[1.0]], dtype=np.float64)
        y = np.array([[2.0]], dtype=np.float64)
        preds, loss = model(x, y)
        assert loss > 0
        assert np.isfinite(loss)
        assert preds.shape == (1, 1)

    def test_training_reduces_loss(self):
        from train_any_llm import NumpyDemoModel

        model = NumpyDemoModel(learning_rate=0.05)
        x = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
        y = np.array([[2.0], [4.0], [6.0]], dtype=np.float64)

        _, loss1 = model(x, y)
        final_loss = loss1
        for _ in range(20):
            _, loss = model(x, y)
            final_loss = loss
        assert final_loss <= loss1 * 1.1 or final_loss < 0.5, "Training should stabilize or reduce loss"

    def test_state_dict(self):
        from train_any_llm import NumpyDemoModel

        model = NumpyDemoModel(learning_rate=0.01)
        state = model.state_dict()
        assert "w" in state
        assert "b" in state
        assert "lr" in state
        assert state["lr"] == 0.01

    def test_load_state_dict(self):
        from train_any_llm import NumpyDemoModel

        model = NumpyDemoModel(learning_rate=0.01)
        original_w = float(model.w.item())
        model.load_state_dict({"w": 0.5, "b": 0.1, "lr": 0.001})
        assert float(model.w) == 0.5
        assert float(model.b) == 0.1
        assert model.lr == 0.001

    def test_convergence_on_linear_data(self):
        """The model should learn a simple linear relationship."""
        from train_any_llm import NumpyDemoModel

        model = NumpyDemoModel(learning_rate=0.1)
        x = np.array([[0.5], [1.0], [1.5], [2.0], [2.5]], dtype=np.float64)
        y = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float64)

        losses = []
        for _ in range(200):
            _, loss = model(x, y)
            losses.append(float(loss))

        assert losses[-1] < losses[0] * 0.5, "Loss should decrease significantly"
        assert losses[-1] < 1.0, "Final loss should be low"


class TestTrainer:
    """Tests for the Trainer class."""

    def test_encode_numpy(self):
        from train_any_llm import NumpyDemoModel, Trainer

        model = NumpyDemoModel()
        trainer = Trainer(model=model)
        encoded = trainer.encode("hello world")
        assert encoded is not None
        assert len(encoded) == 1
        assert encoded[0] > 0

    def test_step_reduces_loss(self):
        import numpy as np

        from train_any_llm import NumpyDemoModel, Trainer

        model = NumpyDemoModel(learning_rate=0.5)
        trainer = Trainer(model=model)

        x = np.array([[1.0]], dtype=np.float64)
        y = np.array([[2.0]], dtype=np.float64)
        loss1 = trainer.step(x, y)
        loss2 = trainer.step(x, y)
        assert loss2 <= loss1 or abs(loss2 - loss1) < 0.01

    def test_get_lr(self):
        from train_any_llm import NumpyDemoModel, Trainer

        model = NumpyDemoModel(learning_rate=0.05)
        trainer = Trainer(model=model)
        assert trainer.get_lr() == 0.05

    def test_encode_empty_text(self):
        from train_any_llm import NumpyDemoModel, Trainer

        model = NumpyDemoModel()
        trainer = Trainer(model=model)
        encoded = trainer.encode("")
        assert encoded is not None
        assert encoded[0] == 0.0


class TestTrainingFunction:
    """Tests for the train() function."""

    def test_training_with_curriculum(self):
        from curriculum import Scheduler
        from train_any_llm import create_demo_model, train

        model = create_demo_model()
        from train_any_llm import Trainer
        trainer = Trainer(model=model)

        curriculum = {
            "easy": ["a" * 20, "b" * 25],
            "medium": ["c" * 50, "d" * 60],
            "hard": ["e" * 100, "f" * 120],
        }
        scheduler = Scheduler(curriculum)

        result = train(model, trainer, scheduler, steps=50)
        assert result.steps_completed == 50
        assert result.training_time >= 0
        assert result.val_bpb > 0
        assert isinstance(result.converged, bool)

    def test_training_result_fields(self):
        from train_any_llm import TrainingResult

        result = TrainingResult(
            val_bpb=0.95,
            training_loss=0.1,
            eval_loss=0.12,
            steps_completed=100,
            training_time=5.0,
            converged=True,
        )
        assert result.val_bpb == 0.95
        assert result.training_loss == 0.1
        assert result.eval_loss == 0.12
        assert result.steps_completed == 100
        assert result.training_time == 5.0
        assert result.converged is True


class TestDemoTraining:
    """Tests for the demo_training() function."""

    def test_demo_training_runs(self):
        from train_any_llm import demo_training

        result = demo_training()
        assert result is not None
        assert result.steps_completed == 100
        assert isinstance(result.converged, bool)
        # Demo should converge (simple linear problem)
        assert result.training_loss < 1.0


class TestCreateDemoModel:
    """Tests for factory function."""

    def test_create(self):
        from train_any_llm import create_demo_model

        model = create_demo_model()
        assert model.lr == 0.01
