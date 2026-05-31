"""Property-based style tests for core modules.

Uses randomized inputs to verify invariant properties:
- Curriculum scheduler: stages progress monotonically
- SimpleVectorStore: add/search work correctly for random data
- Feedback: reward is consistent
"""

import random


class TestCurriculumProperties:
    """Invariant properties for curriculum scheduler.

    Note: Scheduler.get_stage() always returns one of three hardcoded
    stages ("easy", "medium", "hard") regardless of the curriculum dict.
    These properties verify the monotonic progression through those stages.
    """

    STAGE_ORDER = ["easy", "medium", "hard"]

    def test_stages_are_monotonic(self):
        """Stage never decreases as training progresses."""
        from curriculum import Scheduler

        curriculum = {"easy": ["a"], "medium": ["b"], "hard": ["c"]}
        scheduler = Scheduler(curriculum)
        prev_idx = -1
        for step in range(1000):
            stage = scheduler.get_stage(step, 1000)
            idx = self.STAGE_ORDER.index(stage)
            assert idx >= prev_idx
            prev_idx = idx

    def test_first_step_is_first_stage(self):
        """Step 0 is always in 'easy'."""
        from curriculum import Scheduler

        scheduler = Scheduler({"easy": ["a"], "hard": ["c"]})
        for total in [100, 500, 1000]:
            assert scheduler.get_stage(0, total) == "easy"

    def test_last_step_is_last_stage(self):
        """Final step is always in 'hard'."""
        from curriculum import Scheduler

        scheduler = Scheduler({"easy": ["a"], "medium": ["b"], "hard": ["c"]})
        for total in [100, 500, 1000]:
            assert scheduler.get_stage(total - 1, total) == "hard"

    def test_stage_distribution_randomized(self):
        """Run 100 randomized params — properties hold."""
        from curriculum import Scheduler

        rng = random.Random(42)
        for _ in range(100):
            total = rng.randint(10, 2000)
            scheduler = Scheduler({"easy": ["a"], "medium": ["b"], "hard": ["c"]})
            for step in [0, total // 4, total // 2, 3 * total // 4, total - 1]:
                stage = scheduler.get_stage(step, total)
                assert stage in self.STAGE_ORDER

    def test_more_steps_never_backwards(self):
        """Stage never regresses across random progressions."""
        from curriculum import Scheduler

        rng = random.Random(123)
        for _ in range(50):
            total = rng.randint(20, 500)
            scheduler = Scheduler({"easy": ["a"], "medium": ["b"], "hard": ["c"]})
            prev_idx = -1
            for step in range(0, total, max(1, total // 20)):
                stage = scheduler.get_stage(step, total)
                idx = self.STAGE_ORDER.index(stage)
                assert idx >= prev_idx
                prev_idx = idx

    def test_zero_total_returns_medium(self):
        """Edge case: total=0 returns 'medium'."""
        from curriculum import Scheduler

        scheduler = Scheduler({"easy": ["a"], "medium": ["b"], "hard": ["c"]})
        assert scheduler.get_stage(0, 0) == "medium"
        assert scheduler.get_stage(100, 0) == "medium"

    def test_stage_boundaries(self):
        """Verify stage transition points."""
        from curriculum import Scheduler

        scheduler = Scheduler({"easy": ["a"], "medium": ["b"], "hard": ["c"]})
        total = 100
        # 0-32% → easy
        assert scheduler.get_stage(0, total) == "easy"
        assert scheduler.get_stage(32, total) == "easy"
        # 33-65% → medium
        assert scheduler.get_stage(33, total) == "medium"
        assert scheduler.get_stage(65, total) == "medium"
        # 66-99% → hard
        assert scheduler.get_stage(66, total) == "hard"
        assert scheduler.get_stage(99, total) == "hard"


class TestSimpleVectorStoreProperties:
    """Invariant properties for SimpleVectorStore."""

    def test_add_reflects_count(self):
        """Adding N items means store has N items."""
        from memory import ExperimentMemory, SimpleVectorStore

        store = SimpleVectorStore()
        rng = random.Random(42)
        n = rng.randint(1, 30)
        for i in range(n):
            store.add(
                ExperimentMemory(
                    experiment_id=i + 1,
                    timestamp="2026-01-01",
                    change_description=f"experiment {i}",
                    change_type="optimization",
                    val_bpb_before=1.0,
                    val_bpb_after=0.95,
                    status="kept",
                )
            )
        assert len(store.experiments) == n

    def test_search_returns_matches(self):
        """Searching for a keyword returns experiments containing it."""
        from memory import ExperimentMemory, SimpleVectorStore

        store = SimpleVectorStore()
        rng = random.Random(42)
        for i in range(20):
            desc = rng.choice(
                [
                    "learning rate adjustment",
                    "batch size change",
                    "weight decay tuning",
                    "dropout addition",
                ]
            )
            store.add(
                ExperimentMemory(
                    experiment_id=i + 1,
                    timestamp="2026-01-01",
                    change_description=desc,
                    change_type="optimization",
                    val_bpb_before=1.0,
                    val_bpb_after=0.95,
                    status="kept",
                )
            )
        for keyword in ["learning", "batch", "weight", "dropout"]:
            results = store.search(keyword)
            assert len(results) >= 1

    def test_search_respects_limit(self):
        """Search results never exceed the requested limit."""
        from memory import ExperimentMemory, SimpleVectorStore

        store = SimpleVectorStore()
        for i in range(50):
            store.add(
                ExperimentMemory(
                    experiment_id=i + 1,
                    timestamp="2026-01-01",
                    change_description=f"test experiment number {i}",
                    change_type="optimization",
                    val_bpb_before=1.0,
                    val_bpb_after=0.95,
                    status="kept",
                )
            )
        for limit in [1, 5, 10, 30]:
            results = store.search("test", limit=limit)
            assert len(results) <= limit

    def test_empty_query_returns_recent(self):
        """Empty query returns most recent experiments."""
        from memory import ExperimentMemory, SimpleVectorStore

        store = SimpleVectorStore()
        for i in range(10):
            store.add(
                ExperimentMemory(
                    experiment_id=i + 1,
                    timestamp="2026-01-01",
                    change_description=f"exp {i}",
                    change_type="optimization",
                    val_bpb_before=1.0,
                    val_bpb_after=0.95,
                    status="kept",
                )
            )
        results = store.search("", limit=5)
        assert len(results) == min(5, len(store.experiments))


class TestFeedbackProperties:
    """Invariant properties for feedback reward function."""

    def test_reward_is_finite(self):
        """Reward is always a finite float."""
        from feedback import Feedback

        fb = Feedback()
        rng = random.Random(42)
        for _ in range(100):
            before = rng.uniform(0.1, 10.0)
            after = rng.uniform(0.1, 10.0)
            reward = fb.reward(before, after)
            assert isinstance(reward, (int, float))
            assert reward != float("inf")
            assert reward != float("-inf")

    def test_reward_is_positive_for_improvement(self):
        """When val_bpb decreases (improves), reward should be positive."""
        from feedback import Feedback

        fb = Feedback()
        reward = fb.reward(2.0, 0.0)
        assert reward > 0
