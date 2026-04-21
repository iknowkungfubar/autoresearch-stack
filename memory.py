"""
Memory system for autonomous research.

Features:
- Experiment memory with retrieval
- Vector store (Chroma or simple fallback)
- Semantic search over past experiments
- "What has been tried" retrieval
"""

import json
import os
import sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict


@dataclass
class ExperimentMemory:
    """Single experiment memory record."""

    experiment_id: int
    timestamp: str
    change_description: str
    change_type: str
    val_bpb_before: float
    val_bpb_after: float
    status: str
    failure_classification: Optional[str] = None
    failure_diagnosis: Optional[str] = None
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def improved(self) -> bool:
        return self.status == "kept" and self.val_bpb_after < self.val_bpb_before


class SimpleVectorStore:
    """Simple in-memory vector store (fallback when ChromaDB unavailable)."""

    def __init__(self):
        self.experiments: List[ExperimentMemory] = []
        self.index: Dict[str, List[int]] = defaultdict(list)  # word -> experiment_ids

    def add(self, exp: ExperimentMemory):
        """Add experiment to memory."""
        self.experiments.append(exp)

        # Index by words in description
        words = self._tokenize(exp.change_description)
        for word in words:
            self.index[word].append(exp.experiment_id)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return text.lower().split()

    def search(self, query: str, limit: int = 10) -> List[ExperimentMemory]:
        """Search by keyword matching."""
        query_words = self._tokenize(query)

        if not query_words:
            return self.experiments[-limit:]

        # Score by matches
        scores: Dict[int, int] = defaultdict(int)
        for word in query_words:
            for exp_id in self.index.get(word, []):
                scores[exp_id] += 1

        # Sort by score
        scored = sorted(
            [(exp_id, score) for exp_id, score in scores.items()],
            key=lambda x: (-x[1], x[0]),
        )

        # Get experiments
        exp_by_id = {e.experiment_id: e for e in self.experiments}
        results = []
        for exp_id, score in scored[:limit]:
            if exp_id in exp_by_id:
                results.append(exp_by_id[exp_id])

        return results

    def get_by_type(self, change_type: str) -> List[ExperimentMemory]:
        """Get experiments by type."""
        return [e for e in self.experiments if e.change_type == change_type]

    def get_by_status(self, status: str) -> List[ExperimentMemory]:
        """Get experiments by status."""
        return [e for e in self.experiments if e.status == status]

    def get_recent(self, n: int = 10) -> List[ExperimentMemory]:
        """Get recent experiments."""
        return self.experiments[-n:]


class MemorySystem:
    """Main memory system with retrieval."""

    def __init__(
        self, db_path: str = "./experiments.db", memory_path: str = "./memory"
    ):
        self.db_path = db_path
        self.memory_path = Path(memory_path)
        self.memory_path.mkdir(exist_ok=True)

        # Try to use ChromaDB, fallback to simple
        self.use_chroma = False
        self.vector_store = None
        self._init_vector_store()

        # Statistics
        self.stats = {
            "total_queries": 0,
            "hits": 0,
            "misses": 0,
        }

    def _init_vector_store(self):
        """Initialize vector store."""
        try:
            import chromadb
            from chromadb.config import Settings

            # Try to create ChromaDB client
            client = chromadb.Client(
                Settings(
                    persist_directory=str(self.memory_path), anonymized_telemetry=False
                )
            )

            # Create or get collection
            try:
                self.collection = client.create_collection("experiments")
            except:
                self.collection = client.get_collection("experiments")

            self.use_chroma = True
            self.vector_store = SimpleVectorStore()  # Backup
            print("Using ChromaDB vector store")
        except ImportError:
            print("ChromaDB not available, using simple vector store")
            self.vector_store = SimpleVectorStore()
        except Exception as e:
            print(f"ChromaDB init failed: {e}, using simple store")
            self.vector_store = SimpleVectorStore()

    def load_from_db(self):
        """Load experiments from SQLite database."""
        if not Path(self.db_path).exists():
            return

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        cursor = conn.execute("""
            SELECT id, timestamp, change_description, change_type,
                   val_bpb_before, val_bpb_after, status,
                   failure_classification, failure_diagnosis
            FROM experiments
            ORDER BY id DESC
            LIMIT 1000
        """)

        for row in cursor.fetchall():
            exp = ExperimentMemory(
                experiment_id=row["id"],
                timestamp=row["timestamp"],
                change_description=row["change_description"],
                change_type=row["change_type"],
                val_bpb_before=row["val_bpb_before"],
                val_bpb_after=row["val_bpb_after"],
                status=row["status"],
                failure_classification=row["failure_classification"],
                failure_diagnosis=row["failure_diagnosis"],
            )
            self.vector_store.add(exp)

        conn.close()

    def query(self, query: str, limit: int = 10) -> List[ExperimentMemory]:
        """Query memory for similar experiments."""
        self.stats["total_queries"] += 1

        results = self.vector_store.search(query, limit)

        if results:
            self.stats["hits"] += 1
        else:
            self.stats["misses"] += 1

        return results

    def get_what_been_tried(self, component: str) -> List[Dict[str, Any]]:
        """Get what has been tried for a component."""
        experiments = self.vector_store.search(component, limit=20)

        return [
            {
                "id": e.experiment_id,
                "change": e.change_description,
                "type": e.change_type,
                "val_bpb_before": e.val_bpb_before,
                "val_bpb_after": e.val_bpb_after,
                "status": e.status,
                "failure": e.failure_classification,
            }
            for e in experiments
        ]

    def get_patterns(self, change_type: Optional[str] = None) -> Dict[str, List[str]]:
        """Analyze patterns in experiments."""
        # Load from db if empty
        if not self.vector_store.experiments:
            self.load_from_db()

        patterns: Dict[str, List[str]] = defaultdict(list)

        for exp in self.vector_store.experiments:
            if change_type and exp.change_type != change_type:
                continue

            if exp.status == "reverted":
                patterns["failed"].append(exp.change_description)
            elif exp.status == "kept":
                patterns["success"].append(exp.change_description)

        return dict(patterns)

    def suggest_next(self, change_type: str) -> str:
        """Suggest what to try next based on history."""
        patterns = self.get_patterns(change_type)

        # What hasn't been tried for this type
        failures = set(patterns.get("failed", []))

        suggestions = {
            "optimization": [
                "Increase learning rate by 10%",
                "Decrease learning rate by 10%",
                "Add learning rate warmup",
                "Adjust weight decay",
                "Change optimizer to AdamW",
            ],
            "architecture": [
                "Add dropout layer",
                "Increase layer dimension",
                "Add attention head",
                "Adjust hidden size",
            ],
            "curriculum": [
                "Adjust stage timing",
                "Increase warmup ratio",
                "Add difficulty scaling",
            ],
            "synthetic": [
                "Increase sample count",
                "Adjust temperature",
                "Change difficulty mix",
            ],
        }

        # Get suggestions for type
        type_suggestions = suggestions.get(change_type, [])

        # Filter out what failed recently
        available = [s for s in type_suggestions if s not in failures]

        if available:
            return available[0]

        return (
            type_suggestions[0]
            if type_suggestions
            else "Try a different optimization approach"
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        patterns = self.get_patterns()

        return {
            "total_experiments": len(self.vector_store.experiments),
            "successful": len(patterns.get("success", [])),
            "failed": len(patterns.get("failed", [])),
            "query_stats": self.stats,
        }


# Convenience function
def get_memory(db_path: str = "./experiments.db") -> MemorySystem:
    """Get memory system instance."""
    return MemorySystem(db_path=db_path)


def what_been_tried(
    component: str, db_path: str = "./experiments.db"
) -> List[Dict[str, Any]]:
    """Convenience function to query what has been tried."""
    memory = get_memory(db_path)
    memory.load_from_db()
    return memory.get_what_been_tried(component)


if __name__ == "__main__":
    # Demo
    memory = get_memory()
    print(f"Memory system initialized: {type(memory.vector_store).__name__}")

    # Try to load from DB
    if Path("./experiments.db").exists():
        memory.load_from_db()
        stats = memory.get_statistics()
        print(f"Loaded {stats['total_experiments']} experiments")

    # Test query
    if memory.vector_store.experiments:
        results = memory.query("learning rate")
        print(f"Query 'learning rate': {len(results)} results")
