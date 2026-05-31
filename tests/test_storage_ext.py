"""Extended tests for the storage module.

Covers:
- ExperimentDB CRUD operations edge cases
- get_experiments filtering, pagination
- ExperimentJSONL fallback
- Error handling
"""


class TestExperimentDBExtended:
    """Extended tests for ExperimentDB."""

    def test_insert_and_retrieve(self, tmp_path):
        from storage import ExperimentDB

        db = ExperimentDB(str(tmp_path / "test.db"))
        exp_id = db.insert_experiment(
            timestamp="2026-01-01",
            change_description="Test change",
            change_code="code",
            change_type="optimization",
            val_bpb_before=1.0,
            status="running",
        )
        assert exp_id > 0

        exp = db.get_experiment(exp_id)
        assert exp is not None
        assert exp["change_description"] == "Test change"
        assert exp["status"] == "running"

    def test_get_nonexistent(self, tmp_path):
        from storage import ExperimentDB

        db = ExperimentDB(str(tmp_path / "test.db"))
        exp = db.get_experiment(999)
        assert exp is None

    def test_get_experiments_filter_by_status(self, tmp_path):
        from storage import ExperimentDB

        db = ExperimentDB(str(tmp_path / "test.db"))
        for i in range(5):
            db.insert_experiment(
                timestamp="2026-01-01",
                change_description=f"Test {i}",
                change_code="code",
                change_type="optimization",
                val_bpb_before=1.0,
                status="kept" if i % 2 == 0 else "reverted",
            )

        kept = db.get_experiments(status="kept")
        assert len(kept) == 3

        reverted = db.get_experiments(status="reverted")
        assert len(reverted) == 2

    def test_get_experiments_pagination(self, tmp_path):
        from storage import ExperimentDB

        db = ExperimentDB(str(tmp_path / "test.db"))
        for i in range(20):
            db.insert_experiment(
                timestamp="2026-01-01",
                change_description=f"Test {i}",
                change_code="code",
                change_type="optimization",
                val_bpb_before=1.0,
                status="kept",
            )

        page1 = db.get_experiments(limit=5, offset=0)
        assert len(page1) == 5

        page2 = db.get_experiments(limit=5, offset=5)
        assert len(page2) == 5
        assert page1[0]["id"] != page2[0]["id"]

    def test_update_partial(self, tmp_path):
        from storage import ExperimentDB

        db = ExperimentDB(str(tmp_path / "test.db"))
        exp_id = db.insert_experiment(
            timestamp="2026-01-01",
            change_description="Test",
            change_code="code",
            change_type="optimization",
            val_bpb_before=1.0,
            status="running",
        )

        db.update_experiment(exp_id, val_bpb_after=0.95, status="kept")
        exp = db.get_experiment(exp_id)
        assert exp is not None
        assert exp["val_bpb_after"] == 0.95
        assert exp["status"] == "kept"

    def test_update_with_all_fields(self, tmp_path):
        from storage import ExperimentDB

        db = ExperimentDB(str(tmp_path / "test.db"))
        exp_id = db.insert_experiment(
            timestamp="2026-01-01",
            change_description="Test",
            change_code="code",
            change_type="optimization",
            val_bpb_before=1.0,
            status="running",
        )

        db.update_experiment(
            exp_id,
            val_bpb_after=0.9,
            training_loss=0.1,
            eval_loss=0.12,
            training_time=5.0,
            memory_used=1024,
            status="kept",
            failure_classification=None,
            failure_diagnosis=None,
            git_commit="abc123",
            notes="Test notes",
        )
        exp = db.get_experiment(exp_id)
        assert exp is not None
        assert exp["val_bpb_after"] == 0.9
        assert exp["training_loss"] == 0.1
        assert exp["training_time"] == 5.0

    def test_multiple_updates(self, tmp_path):
        from storage import ExperimentDB

        db = ExperimentDB(str(tmp_path / "test.db"))
        exp_id = db.insert_experiment(
            timestamp="2026-01-01",
            change_description="Test",
            change_code="code",
            change_type="optimization",
            val_bpb_before=1.0,
            status="running",
        )

        db.update_experiment(exp_id, val_bpb_after=0.95, status="kept")
        db.update_experiment(exp_id, val_bpb_after=0.90, status="kept")
        exp = db.get_experiment(exp_id)
        assert exp is not None
        assert exp["val_bpb_after"] == 0.90

    def test_get_statistics_structure(self, tmp_path):
        from storage import ExperimentDB

        db = ExperimentDB(str(tmp_path / "test.db"))
        for i in range(5):
            db.insert_experiment(
                timestamp="2026-01-01",
                change_description=f"Test {i}",
                change_code="code",
                change_type="optimization",
                val_bpb_before=1.0,
                status="kept" if i < 3 else "reverted",
            )
            if i < 3:
                db.update_experiment(i + 1, val_bpb_after=0.95, status="kept")
            else:
                db.update_experiment(i + 1, val_bpb_after=1.2, status="reverted")

        stats = db.get_statistics()
        assert stats["total_experiments"] == 5


class TestExperimentJSONL:
    """Tests for ExperimentJSONL fallback storage."""

    def test_init_and_append(self, tmp_path):
        from storage import ExperimentJSONL

        log_path = str(tmp_path / "experiments.jsonl")
        store = ExperimentJSONL(log_path=log_path)
        assert store is not None

        store.append({"id": 1, "name": "test"})
        store.append({"id": 2, "name": "test2"})

        assert len(store.experiments) == 2

    def test_persistence(self, tmp_path):
        from storage import ExperimentJSONL

        log_path = str(tmp_path / "experiments.jsonl")

        store1 = ExperimentJSONL(log_path=log_path)
        store1.append({"id": 1, "name": "test"})
        del store1

        store2 = ExperimentJSONL(log_path=log_path)
        assert len(store2.experiments) == 1
        assert store2.experiments[0]["id"] == 1

    def test_append_multiple(self, tmp_path):
        from storage import ExperimentJSONL

        log_path = str(tmp_path / "experiments.jsonl")
        store = ExperimentJSONL(log_path=log_path)

        for i in range(10):
            store.append({"id": i, "value": f"data_{i}"})

        assert len(store.experiments) == 10

    def test_load_empty_file(self, tmp_path):
        from storage import ExperimentJSONL

        log_path = str(tmp_path / "nonexistent.jsonl")
        store = ExperimentJSONL(log_path=log_path)
        assert len(store.experiments) == 0

    def test_corrupted_line_skipped(self, tmp_path):
        from storage import ExperimentJSONL

        log_path = tmp_path / "experiments.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            '{"id": 1, "name": "valid"}\n'
            "not valid json\n"
            '{"id": 2, "name": "valid2"}\n'
        )

        store = ExperimentJSONL(log_path=str(log_path))
        assert len(store.experiments) == 2
