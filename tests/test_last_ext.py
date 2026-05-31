"""Extended tests for checkpoint, daemon, and distribute modules."""
import pytest


class TestCheckpointExtended:
    def test_save_and_load_progress(self, tmp_path):
        from checkpoint import CheckpointManager
        mgr = CheckpointManager(str(tmp_path / "ckpts"))
        ckpt_id = mgr.save_progress(experiment_id=1, iteration=50, val_bpb=0.95)
        assert ckpt_id is not None
        loaded = mgr.load(ckpt_id)
        assert loaded is not None


class TestDaemonExtended:
    def test_save_and_load_stats_with_data(self, tmp_path):
        from daemon import Daemon, DaemonConfig
        import json
        sf = tmp_path / "s.json"
        sf.write_text(json.dumps({"experiments_run": 10}))
        d = Daemon(DaemonConfig(stats_file=str(sf), log_file=str(tmp_path / "t.log"), pid_file=str(tmp_path / "p.pid")))
        assert d._load_stats()["experiments_run"] == 10


class TestDistributeExtended:
    def test_cluster_add_node(self):
        from distribute import Cluster, NodeConfig, NodeRole
        c = Cluster()
        n = c.add_node(NodeConfig(role=NodeRole.WORKER, name="w1"))
        assert n.id == "worker-w1"

    def test_cluster_multiple_nodes(self):
        from distribute import Cluster, NodeConfig
        c = Cluster()
        c.add_node(NodeConfig(name="n1"))
        c.add_node(NodeConfig(name="n2"))
        c.add_node(NodeConfig(name="n3"))
        assert len(c.nodes) == 3

    def test_cluster_get_workers(self):
        from distribute import Cluster, NodeConfig, NodeRole
        c = Cluster()
        c.add_node(NodeConfig(role=NodeRole.MASTER, name="master"))
        c.add_node(NodeConfig(role=NodeRole.WORKER, name="w1"))
        c.add_node(NodeConfig(role=NodeRole.WORKER, name="w2"))
        assert len(c.get_workers()) == 2

    def test_cluster_total_resources(self):
        from distribute import Cluster, NodeConfig
        c = Cluster()
        from distribute import NodeConfig as NC
        c.add_node(NC(cpu_cores=8, memory_gb=32))
        c.add_node(NC(cpu_cores=4, memory_gb=16))
        res = c.total_resources()
        assert res["cpu_cores"] >= 4
        assert res["memory_gb"] >= 16

    def test_cluster_remove_node(self):
        from distribute import Cluster, NodeConfig
        c = Cluster()
        n = c.add_node(NodeConfig(name="test"))
        assert c.remove_node(n.id) is True
        assert c.remove_node("nonexistent") is False

    def test_cluster_to_dict(self):
        from distribute import Cluster, NodeConfig
        c = Cluster()
        c.add_node(NodeConfig(name="n1"))
        d = c.to_dict()
        assert "name" in d
        assert "nodes" in d
        assert "total_resources" in d

    def test_cluster_least_loaded_worker(self):
        from distribute import Cluster, NodeConfig, NodeRole
        c = Cluster()
        n1 = c.add_node(NodeConfig(role=NodeRole.WORKER, name="w1"))
        n2 = c.add_node(NodeConfig(role=NodeRole.WORKER, name="w2"))
        n1.status = "running"
        n2.status = "running"
        n1.metrics.cpu_percent = 80
        n2.metrics.cpu_percent = 20
        least = c.get_least_loaded_worker()
        assert least is not None
        assert least.id == "worker-w2"

    def test_resource_manager_init(self):
        from distribute import ResourceManager, Cluster
        rm = ResourceManager(cluster=Cluster())
        assert rm is not None

    def test_cost_estimator_init(self):
        from distribute import CostEstimator
        ce = CostEstimator()
        assert ce is not None
