"""Extended tests for the memory module.

Covers:
- MemorySystem initialization and ChromaDB fallback
- load_from_db with empty/missing database
- Pattern analysis (get_patterns, get_what_been_tried)
- suggest_next with various scenarios
- Statistics tracking
"""


class TestMemorySystem:
    """Tests for MemorySystem."""

    def test_init_fallback_to_simple(self, tmp_path):
        """Without ChromaDB, falls back to SimpleVectorStore."""
        from memory import MemorySystem

        memory = MemorySystem(db_path=str(tmp_path / "test.db"))
        assert memory.vector_store is not None
        assert memory.use_chroma is False

    def test_init_with_memory_path(self, tmp_path):
        """Memory path is created if it doesn't exist."""
        from memory import MemorySystem

        mem_path = tmp_path / "test_memory"
        memory = MemorySystem(
            db_path=str(tmp_path / "test.db"),
            memory_path=str(mem_path),
        )
        assert mem_path.exists()

    def test_load_from_db_missing(self, tmp_path):
        """Loading from nonexistent DB is handled gracefully."""
        from memory import MemorySystem

        memory = MemorySystem(db_path=str(tmp_path / "nonexistent.db"))
        # Should not raise
        memory.load_from_db()
        assert len(memory.vector_store.experiments) == 0

    def test_get_what_been_tried_empty(self, tmp_path):
        """Empty memory returns empty list."""
        from memory import MemorySystem

        memory = MemorySystem(db_path=str(tmp_path / "test.db"))
        result = memory.get_what_been_tried("learning rate")
        assert result == []

    def test_get_patterns_empty(self, tmp_path):
        """Empty memory returns empty pattern dicts."""
        from memory import MemorySystem

        memory = MemorySystem(db_path=str(tmp_path / "test.db"))
        patterns = memory.get_patterns()
        assert isinstance(patterns, dict)

    def test_get_patterns_filtered(self, tmp_path):
        """Filtering by change_type works."""
        from memory import ExperimentMemory, MemorySystem

        memory = MemorySystem(db_path=str(tmp_path / "test.db"))
        store = memory.vector_store

        store.add(
            ExperimentMemory(
                experiment_id=1,
                timestamp="2026-01-01",
                change_description="LR test",
                change_type="optimization",
                val_bpb_before=1.0,
                val_bpb_after=0.95,
                status="kept",
            )
        )
        store.add(
            ExperimentMemory(
                experiment_id=2,
                timestamp="2026-01-01",
                change_description="Dropout test",
                change_type="architecture",
                val_bpb_before=1.0,
                val_bpb_after=1.2,
                status="reverted",
            )
        )

        patterns = memory.get_patterns(change_type="optimization")
        assert len(patterns["success"]) >= 1
        assert "LR test" in patterns["success"]

    def test_suggest_next_unknown_type(self, tmp_path):
        """suggest_next returns a fallback for unknown change types."""
        from memory import MemorySystem

        memory = MemorySystem(db_path=str(tmp_path / "test.db"))
        suggestion = memory.suggest_next("unknown_type")
        assert isinstance(suggestion, str)
        assert len(suggestion) > 0

    def test_suggest_next_all_failed(self, tmp_path):
        """When all suggestions have failed, returns first anyway."""
        from memory import ExperimentMemory, MemorySystem

        memory = MemorySystem(db_path=str(tmp_path / "test.db"))
        store = memory.vector_store

        # Add many failed optimization experiments
        for i in range(5):
            store.add(
                ExperimentMemory(
                    experiment_id=i + 1,
                    timestamp="2026-01-01",
                    change_description=[
                        "Increase learning rate by 10%",
                        "Decrease learning rate by 10%",
                        "Add learning rate warmup",
                        "Adjust weight decay",
                        "Change optimizer to AdamW",
                    ][i],
                    change_type="optimization",
                    val_bpb_before=1.0,
                    val_bpb_after=1.5,
                    status="reverted",
                )
            )

        suggestion = memory.suggest_next("optimization")
        assert isinstance(suggestion, str)
        assert len(suggestion) > 0

    def test_query_hits_and_misses(self, tmp_path):
        """Query statistics track hits and misses."""
        from memory import ExperimentMemory, MemorySystem

        memory = MemorySystem(db_path=str(tmp_path / "test.db"))
        store = memory.vector_store

        store.add(
            ExperimentMemory(
                experiment_id=1,
                timestamp="2026-01-01",
                change_description="learning rate test",
                change_type="optimization",
                val_bpb_before=1.0,
                val_bpb_after=0.95,
                status="kept",
            )
        )

        # Hit
        memory.query("learning")
        # Miss
        memory.query("nonexistent_keyword_xyz")

        assert memory.stats["total_queries"] == 2
        assert memory.stats["hits"] >= 1
        assert memory.stats["misses"] >= 1

    def test_get_statistics(self, tmp_path):
        """get_statistics returns structured data."""
        from memory import ExperimentMemory, MemorySystem

        memory = MemorySystem(db_path=str(tmp_path / "test.db"))
        store = memory.vector_store

        store.add(
            ExperimentMemory(
                experiment_id=1,
                timestamp="2026-01-01",
                change_description="Good change",
                change_type="optimization",
                val_bpb_before=1.0,
                val_bpb_after=0.95,
                status="kept",
            )
        )
        store.add(
            ExperimentMemory(
                experiment_id=2,
                timestamp="2026-01-01",
                change_description="Bad change",
                change_type="optimization",
                val_bpb_before=1.0,
                val_bpb_after=1.5,
                status="reverted",
            )
        )

        stats = memory.get_statistics()
        assert stats["total_experiments"] == 2
        assert stats["successful"] >= 1
        assert stats["failed"] >= 1


class TestExperimentMemory:
    """Tests for ExperimentMemory dataclass."""

    def test_improved_property(self):
        from memory import ExperimentMemory

        exp = ExperimentMemory(
            experiment_id=1,
            timestamp="2026-01-01",
            change_description="Test",
            change_type="opt",
            val_bpb_before=1.0,
            val_bpb_after=0.9,
            status="kept",
        )
        assert exp.improved is True

    def test_not_improved(self):
        from memory import ExperimentMemory

        exp = ExperimentMemory(
            experiment_id=1,
            timestamp="2026-01-01",
            change_description="Test",
            change_type="opt",
            val_bpb_before=1.0,
            val_bpb_after=1.1,
            status="reverted",
        )
        assert exp.improved is False

    def test_to_dict(self):
        from memory import ExperimentMemory

        exp = ExperimentMemory(
            experiment_id=1,
            timestamp="2026-01-01",
            change_description="Test",
            change_type="opt",
            val_bpb_before=1.0,
            val_bpb_after=0.95,
            status="kept",
        )
        d = exp.to_dict()
        assert d["experiment_id"] == 1
        assert d["change_description"] == "Test"

    def test_tags_default(self):
        from memory import ExperimentMemory

        exp = ExperimentMemory(
            experiment_id=1,
            timestamp="2026-01-01",
            change_description="Test",
            change_type="opt",
            val_bpb_before=1.0,
            val_bpb_after=0.95,
            status="kept",
        )
        assert exp.tags == []


class TestSimpleVectorStoreExtended:
    """Extended tests for SimpleVectorStore."""

    def test_get_by_type(self):
        from memory import ExperimentMemory, SimpleVectorStore

        store = SimpleVectorStore()
        store.add(
            ExperimentMemory(
                experiment_id=1,
                timestamp="2026-01-01",
                change_description="LR",
                change_type="optimization",
                val_bpb_before=1.0,
                val_bpb_after=0.95,
                status="kept",
            )
        )
        store.add(
            ExperimentMemory(
                experiment_id=2,
                timestamp="2026-01-01",
                change_description="Dropout",
                change_type="architecture",
                val_bpb_before=1.0,
                val_bpb_after=1.2,
                status="reverted",
            )
        )

        opts = store.get_by_type("optimization")
        assert len(opts) == 1
        assert opts[0].change_type == "optimization"

    def test_get_by_status(self):
        from memory import ExperimentMemory, SimpleVectorStore

        store = SimpleVectorStore()
        store.add(
            ExperimentMemory(
                experiment_id=1,
                timestamp="2026-01-01",
                change_description="Kept",
                change_type="opt",
                val_bpb_before=1.0,
                val_bpb_after=0.95,
                status="kept",
            )
        )
        store.add(
            ExperimentMemory(
                experiment_id=2,
                timestamp="2026-01-01",
                change_description="Reverted",
                change_type="opt",
                val_bpb_before=1.0,
                val_bpb_after=1.2,
                status="reverted",
            )
        )

        kept = store.get_by_status("kept")
        assert len(kept) == 1
        assert kept[0].status == "kept"

    def test_get_recent(self):
        from memory import ExperimentMemory, SimpleVectorStore

        store = SimpleVectorStore()
        for i in range(10):
            store.add(
                ExperimentMemory(
                    experiment_id=i + 1,
                    timestamp="2026-01-01",
                    change_description=f"exp {i}",
                    change_type="opt",
                    val_bpb_before=1.0,
                    val_bpb_after=0.95,
                    status="kept",
                )
            )

        recent = store.get_recent(3)
        assert len(recent) == 3

    def test_get_recent_more_than_available(self):
        from memory import ExperimentMemory, SimpleVectorStore

        store = SimpleVectorStore()
        store.add(
            ExperimentMemory(
                experiment_id=1,
                timestamp="2026-01-01",
                change_description="test",
                change_type="opt",
                val_bpb_before=1.0,
                val_bpb_after=0.95,
                status="kept",
            )
        )
        recent = store.get_recent(100)
        assert len(recent) == 1

    def test_search_no_keyword_matches(self):
        from memory import ExperimentMemory, SimpleVectorStore

        store = SimpleVectorStore()
        store.add(
            ExperimentMemory(
                experiment_id=1,
                timestamp="2026-01-01",
                change_description="learning rate experiment",
                change_type="opt",
                val_bpb_before=1.0,
                val_bpb_after=0.95,
                status="kept",
            )
        )
        results = store.search("zzz_nonexistent")
        assert len(results) == 0

    def test_index_updates_on_add(self):
        from memory import ExperimentMemory, SimpleVectorStore

        store = SimpleVectorStore()
        assert len(store.index) == 0

        store.add(
            ExperimentMemory(
                experiment_id=1,
                timestamp="2026-01-01",
                change_description="learning rate test",
                change_type="opt",
                val_bpb_before=1.0,
                val_bpb_after=0.95,
                status="kept",
            )
        )
        assert "learning" in store.index
        assert "rate" in store.index
