"""Extended tests for autonomous loop, distribute, and metaloop modules."""


class TestAutonomousPipelineExtended:
    """More autonomous_loop tests."""

    def test_autonomous_pipeline_from_texts(self):
        from autoresearch.experiment.autonomous_loop import autonomous_pipeline

        data = autonomous_pipeline(
            raw=["machine learning is a method", "neural networks are powerful"],
            config_path="config.yaml",
        )
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_pipeline_init_sets_best_val_bpb(self):
        from autoresearch.experiment.autonomous_loop import AutonomousPipeline

        p = AutonomousPipeline("config.yaml")
        assert p.best_val_bpb == float("inf")


class TestMetaLoopExtended:
    """Extended metaloop tests."""

    def test_register_prompt(self):
        from autoresearch.infrastructure.metaloop import MetaLoop

        m = MetaLoop()
        p = m.register_prompt("test", "content")
        assert p.name == "test"
        assert p.version >= 1

    def test_evolve_heuristic(self):
        from autoresearch.infrastructure.metaloop import MetaLoop

        m = MetaLoop()
        m.register_prompt("test", "initial prompt")
        evolved = m.evolve_prompt("test", "vague feedback", 0.9)
        assert evolved.version >= 2
        assert "specific" in evolved.content.lower()

    def test_evolve_with_constraint_feedback(self):
        from autoresearch.infrastructure.metaloop import MetaLoop

        m = MetaLoop()
        m.register_prompt("test", "do stuff")
        evolved = m.evolve_prompt("test", "unconstrained approach", 0.8)
        assert "constraint" in evolved.content.lower()

    def test_propose_hyperparameter_change(self):
        from autoresearch.infrastructure.metaloop import MetaLoop

        m = MetaLoop()
        mod = m.propose_hyperparameter_change("lr", 0.01, "increase")
        assert mod.type.value == "hyperparameter"
        assert "lr" in mod.description

    def test_modification_lifecycle(self):
        from autoresearch.infrastructure.metaloop import MetaLoop

        m = MetaLoop()
        mod = m.propose_hyperparameter_change("bs", 32, "increase")
        m.apply_modification(mod.id)
        assert any(mo.status == "applied" for mo in m.modifications if mo.id == mod.id)
        m.revert_modification(mod.id)
        assert any(mo.status == "reverted" for mo in m.modifications if mo.id == mod.id)

    def test_record_impact(self):
        from autoresearch.infrastructure.metaloop import MetaLoop

        m = MetaLoop()
        mod = m.propose_hyperparameter_change("lr", 0.01, "increase")
        m.record_impact(mod.id, 0.05)
        assert any(
            mo.actual_impact == 0.05 for mo in m.modifications if mo.id == mod.id
        )

    def test_analyze_patterns(self):
        from autoresearch.infrastructure.metaloop import MetaLoop

        m = MetaLoop()
        result = m.analyze_patterns()
        assert isinstance(result, dict)


class TestDistributeExtended:
    """Extended distribute tests."""

    def test_node_config_default_name(self):
        from autoresearch.infrastructure.distribute import NodeConfig

        c = NodeConfig()
        assert c.name == "worker-1"

    def test_node_creation_defaults(self):
        from autoresearch.infrastructure.distribute import Node, NodeConfig

        n = Node(NodeConfig())
        assert n.status == "pending"
        assert n.experiments_completed == 0

    def test_node_metrics_update(self):
        from autoresearch.infrastructure.distribute import Node, NodeConfig

        n = Node(NodeConfig())
        n.metrics.cpu_percent = 75.0
        assert n.metrics.cpu_percent == 75.0

    def test_node_health_boundaries(self):
        from autoresearch.infrastructure.distribute import Node, NodeConfig

        n = Node(NodeConfig())
        n.status = "running"
        n.metrics.cpu_percent = 80
        n.metrics.memory_percent = 80
        assert n.is_healthy() is True

        n.metrics.cpu_percent = 90
        assert n.is_healthy() is False

    def test_cost_estimate_default_currency(self):
        from autoresearch.infrastructure.distribute import CloudProvider, CostEstimate

        e = CostEstimate(CloudProvider.AWS, "t3.medium", 0.1, 10, 1.0)
        assert e.currency == "USD"

    def test_cluster_node(self):
        from autoresearch.infrastructure.distribute import Node, NodeConfig

        n = Node(NodeConfig())
        assert isinstance(n.id, str)
