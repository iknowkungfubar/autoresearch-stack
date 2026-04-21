"""
Experiment storage and retrieval.

Provides SQLite-backed experiment database with JSONL fallback.
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import contextmanager


class ExperimentDB:
    """SQLite-based experiment storage."""

    def __init__(self, db_path: str = "./experiments.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    change_description TEXT,
                    change_code TEXT,
                    change_type TEXT,
                    val_bpb_before REAL,
                    val_bpb_after REAL,
                    training_loss REAL,
                    eval_loss REAL,
                    training_time REAL,
                    memory_used REAL,
                    status TEXT NOT NULL,
                    failure_classification TEXT,
                    failure_diagnosis TEXT,
                    git_commit TEXT,
                    notes TEXT,
                    tags TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status ON experiments(status)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON experiments(timestamp)
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER,
                    step INTEGER,
                    state_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                )
            """)

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def insert_experiment(
        self,
        timestamp: str,
        change_description: str,
        change_code: str,
        change_type: str,
        val_bpb_before: float,
        status: str = "running",
    ) -> int:
        """Insert a new experiment."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO experiments (
                    timestamp, change_description, change_code, change_type,
                    val_bpb_before, status
                ) VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    timestamp,
                    change_description,
                    change_code,
                    change_type,
                    val_bpb_before,
                    status,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def update_experiment(
        self,
        experiment_id: int,
        val_bpb_after: Optional[float] = None,
        training_loss: Optional[float] = None,
        eval_loss: Optional[float] = None,
        training_time: Optional[float] = None,
        memory_used: Optional[float] = None,
        status: Optional[str] = None,
        failure_classification: Optional[str] = None,
        failure_diagnosis: Optional[str] = None,
        git_commit: Optional[str] = None,
        notes: Optional[str] = None,
    ):
        """Update experiment with results."""
        updates = []
        values = []

        if val_bpb_after is not None:
            updates.append("val_bpb_after = ?")
            values.append(val_bpb_after)
        if training_loss is not None:
            updates.append("training_loss = ?")
            values.append(training_loss)
        if eval_loss is not None:
            updates.append("eval_loss = ?")
            values.append(eval_loss)
        if training_time is not None:
            updates.append("training_time = ?")
            values.append(training_time)
        if memory_used is not None:
            updates.append("memory_used = ?")
            values.append(memory_used)
        if status is not None:
            updates.append("status = ?")
            values.append(status)
        if failure_classification is not None:
            updates.append("failure_classification = ?")
            values.append(failure_classification)
        if failure_diagnosis is not None:
            updates.append("failure_diagnosis = ?")
            values.append(failure_diagnosis)
        if git_commit is not None:
            updates.append("git_commit = ?")
            values.append(git_commit)
        if notes is not None:
            updates.append("notes = ?")
            values.append(notes)

        if updates:
            values.append(experiment_id)
            with self._get_connection() as conn:
                conn.execute(
                    f"""
                    UPDATE experiments
                    SET {", ".join(updates)}
                    WHERE id = ?
                """,
                    values,
                )
                conn.commit()

    def get_experiment(self, experiment_id: int) -> Optional[Dict]:
        """Get experiment by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_experiments(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict]:
        """Get experiments with optional filtering."""
        query = "SELECT * FROM experiments"
        params = []

        if status:
            query += " WHERE status = ?"
            params.append(status)

        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_kept_experiments(self, limit: int = 100) -> List[Dict]:
        """Get experiments that were kept."""
        return self.get_experiments(status="kept", limit=limit)

    def get_reverted_experiments(self, limit: int = 100) -> List[Dict]:
        """Get experiments that were reverted."""
        return self.get_experiments(status="reverted", limit=limit)

    def get_best_val_bpb(self) -> Optional[float]:
        """Get best (lowest) val_bpb from kept experiments."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT MIN(val_bpb_after) as best
                FROM experiments
                WHERE status = 'kept'
            """)
            row = cursor.fetchone()
            return row["best"] if row else None

    def get_statistics(self) -> Dict[str, Any]:
        """Get experiment statistics."""
        with self._get_connection() as conn:
            # Total
            cursor = conn.execute("SELECT COUNT(*) as count FROM experiments")
            total = cursor.fetchone()["count"]

            # By status
            cursor = conn.execute("""
                SELECT status, COUNT(*) as count
                FROM experiments
                GROUP BY status
            """)
            by_status = {row["status"]: row["count"] for row in cursor.fetchall()}

            # Best val_bpb
            best = self.get_best_val_bpb()

            # First kept val_bpb (baseline)
            cursor = conn.execute("""
                SELECT val_bpb_before
                FROM experiments
                WHERE status = 'kept'
                ORDER BY id ASC
                LIMIT 1
            """)
            baseline = cursor.fetchone()
            baseline_val_bpb = baseline["val_bpb_before"] if baseline else None

            return {
                "total_experiments": total,
                "kept": by_status.get("kept", 0),
                "reverted": by_status.get("reverted", 0),
                "running": by_status.get("running", 0),
                "failed": by_status.get("failed", 0),
                "baseline_val_bpb": baseline_val_bpb,
                "best_val_bpb": best,
                "improvement": (
                    baseline_val_bpb - best if baseline_val_bpb and best else None
                ),
            }

    def save_checkpoint(
        self,
        experiment_id: int,
        step: int,
        state: Dict[str, Any],
    ):
        """Save a training checkpoint."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO checkpoints (experiment_id, step, state_json)
                VALUES (?, ?, ?)
            """,
                (experiment_id, step, json.dumps(state)),
            )
            conn.commit()

    def get_latest_checkpoint(
        self,
        experiment_id: int,
    ) -> Optional[Dict]:
        """Get latest checkpoint for an experiment."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM checkpoints
                WHERE experiment_id = ?
                ORDER BY step DESC
                LIMIT 1
            """,
                (experiment_id,),
            )
            row = cursor.fetchone()
            if row:
                d = dict(row)
                d["state"] = json.loads(d["state_json"])
                return d
            return None


class ExperimentJSONL:
    """JSONL-based experiment storage (simple fallback)."""

    def __init__(self, log_path: str = "./experiments.jsonl"):
        self.log_path = Path(log_path)
        self.experiments: List[Dict] = []
        self._load()

    def _load(self):
        """Load experiments from JSONL."""
        if self.log_path.exists():
            with open(self.log_path, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            self.experiments.append(json.loads(line))
                        except Exception:
                            pass

    def append(self, experiment: Dict):
        """Append experiment to log."""
        self.experiments.append(experiment)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(experiment) + "\n")

    def get_all(self) -> List[Dict]:
        """Get all experiments."""
        return self.experiments

    def get_by_status(self, status: str) -> List[Dict]:
        """Get experiments by status."""
        return [e for e in self.experiments if e.get("status") == status]

    def get_best(self) -> Optional[Dict]:
        """Get best kept experiment."""
        kept = self.get_by_status("kept")
        if not kept:
            return None
        return min(kept, key=lambda x: x.get("val_bpb_after", float("inf")))


# Factory function
def get_experiment_storage(
    use_sqlite: bool = True,
    db_path: str = "./experiments.db",
    jsonl_path: str = "./experiments.jsonl",
) -> Any:
    """Get experiment storage instance."""
    if use_sqlite and sqlite3.sqlite_version:
        return ExperimentDB(db_path)
    else:
        return ExperimentJSONL(jsonl_path)


# Convenience functions
def init_storage(config: Optional[Dict] = None) -> ExperimentDB:
    """Initialize storage from config."""
    config = config or {}
    return ExperimentDB(config.get("db_path", "./experiments.db"))
