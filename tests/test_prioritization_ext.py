"""Tests for the prioritization module (bandit algorithms).

Covers:
- BanditSelector with UCB1, epsilon-greedy, Thompson sampling
- Arm statistics and updates
- PrioritizationSystem integration
- Edge cases (empty arms, zero pulls)
"""


class TestBanditSelector:
    """Tests for bandit-based selection algorithms."""

    def test_ucb1_selection(self):
        from prioritization import BanditSelector

        selector = BanditSelector(strategy="ucb1")

        for _ in range(5):
            arm = selector.select("optimization")
            assert arm in [
                "learning_rate",
                "batch_size",
                "weight_decay",
                "warmup_steps",
                "optimizer",
            ]
            selector.update(arm, reward=1.0, category="optimization")

    def test_epsilon_greedy_selection(self):
        from prioritization import BanditSelector

        selector = BanditSelector(strategy="epsilon_greedy", epsilon=0.5)

        for _ in range(5):
            arm = selector.select("optimization")
            assert arm is not None
            selector.update(arm, reward=1.0, category="optimization")

    def test_thompson_selection(self):
        from prioritization import BanditSelector

        selector = BanditSelector(strategy="thompson")

        for _ in range(5):
            arm = selector.select("optimization")
            assert arm is not None
            selector.update(arm, reward=1.0, category="optimization")

    def test_multiple_categories(self):
        from prioritization import BanditSelector

        selector = BanditSelector(strategy="ucb1")

        for cat in ["optimization", "architecture", "curriculum"]:
            for _ in range(3):
                arm = selector.select(cat)
                assert arm is not None
                selector.update(arm, reward=1.0, category=cat)

    def test_get_statistics(self):
        from prioritization import BanditSelector

        selector = BanditSelector(strategy="ucb1")

        selector.update("learning_rate", reward=1.0, category="optimization")
        selector.update("batch_size", reward=0.5, category="optimization")
        selector.update("learning_rate", reward=0.8, category="optimization")

        stats = selector.get_statistics("optimization")
        assert "learning_rate" in stats
        assert "batch_size" in stats
        assert stats["learning_rate"]["pulls"] == 2
        assert stats["learning_rate"]["mean_reward"] == 0.9

    def test_statistics_empty_category(self):
        from prioritization import BanditSelector

        selector = BanditSelector(strategy="ucb1")
        stats = selector.get_statistics("nonexistent")
        assert stats == {}

    def test_negative_reward(self):
        from prioritization import BanditSelector

        selector = BanditSelector(strategy="ucb1")
        selector.update("learning_rate", reward=-0.5, category="optimization")
        stats = selector.get_statistics("optimization")
        assert stats["learning_rate"]["mean_reward"] == -0.5

    def test_zero_reward(self):
        from prioritization import BanditSelector

        selector = BanditSelector(strategy="ucb1")
        selector.update("learning_rate", reward=0.0, category="optimization")
        stats = selector.get_statistics("optimization")
        assert stats["learning_rate"]["pulls"] == 1

    def test_random_category_selection(self):
        """When category is None, picks a random existing category."""
        from prioritization import BanditSelector

        selector = BanditSelector(strategy="ucb1")
        selector.update("learning_rate", reward=1.0, category="optimization")

        arm = selector.select(category=None)
        assert arm is not None


class TestPrioritizationSystem:
    """Tests for the PrioritizationSystem class."""

    def test_suggest_next(self):
        from prioritization import get_prioritization

        system = get_prioritization(strategy="ucb1")
        suggestion = system.suggest_next(baseline_val_bpb=1.0)
        assert "category" in suggestion
        assert "change" in suggestion
        assert "reasoning" in suggestion

    def test_record_result(self):
        from prioritization import get_prioritization

        system = get_prioritization(strategy="ucb1")

        system.record_result(
            change="Increase learning rate",
            change_type="optimization",
            val_bpb_before=1.0,
            val_bpb_after=0.95,
        )

        assert len(system.experiment_history) >= 1

    def test_record_result_improvement(self):
        from prioritization import get_prioritization

        system = get_prioritization(strategy="ucb1")

        for _ in range(5):
            system.record_result(
                change="Test change",
                change_type="optimization",
                val_bpb_before=1.0,
                val_bpb_after=0.9,
            )

        suggestion = system.suggest_next(baseline_val_bpb=0.9)
        assert "change" in suggestion

    def test_record_result_regression(self):
        from prioritization import get_prioritization

        system = get_prioritization(strategy="ucb1")

        system.record_result(
            change="Bad change",
            change_type="optimization",
            val_bpb_before=1.0,
            val_bpb_after=1.5,
        )

        suggestion = system.suggest_next(baseline_val_bpb=1.5)
        assert suggestion is not None

    def test_experiment_history(self):
        from prioritization import get_prioritization

        system = get_prioritization(strategy="ucb1")

        for cat in ["optimization", "architecture"]:
            for _ in range(3):
                system.record_result(
                    change=f"Change for {cat}",
                    change_type=cat,
                    val_bpb_before=1.0,
                    val_bpb_after=0.95,
                )

        assert len(system.experiment_history) >= 6

    def test_different_strategies(self):
        from prioritization import get_prioritization

        for strategy in ["ucb1", "epsilon_greedy", "thompson"]:
            system = get_prioritization(strategy=strategy)
            system.record_result("test", "optimization", 1.0, 0.95)
            suggestion = system.suggest_next(baseline_val_bpb=0.95)
            assert suggestion is not None
