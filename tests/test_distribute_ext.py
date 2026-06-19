"""Comprehensive tests for distribute module."""


class TestNodeRole:
    """Tests for NodeRole enum."""

    def test_values(self):
        from autoresearch.infrastructure.distribute import NodeRole

        assert NodeRole.MASTER.value == "master"
        assert NodeRole.WORKER.value == "worker"
        assert NodeRole.SCHEDULER.value == "scheduler"


class TestCloudProvider:
    """Tests for CloudProvider enum."""

    def test_values(self):
        from autoresearch.infrastructure.distribute import CloudProvider

        assert CloudProvider.AWS.value == "aws"
        assert CloudProvider.GCP.value == "gcp"
        assert CloudProvider.AZURE.value == "azure"
        assert CloudProvider.LOCAL.value == "local"


class TestNodeConfig:
    """Tests for NodeConfig dataclass."""

    def test_defaults(self):
        from autoresearch.infrastructure.distribute import NodeConfig, NodeRole

        config = NodeConfig()
        assert config.role == NodeRole.WORKER
        assert config.name == "worker-1"
        assert config.cpu_cores == 4
        assert config.memory_gb == 16
        assert config.gpu_count == 0

    def test_custom(self):
        from autoresearch.infrastructure.distribute import NodeConfig, NodeRole

        config = NodeConfig(
            role=NodeRole.MASTER,
            name="master-1",
            cpu_cores=8,
            memory_gb=32,
            gpu_count=2,
            gpu_type="A100",
        )
        assert config.name == "master-1"
        assert config.gpu_type == "A100"


class TestResourceMetrics:
    """Tests for ResourceMetrics dataclass."""

    def test_defaults(self):
        from autoresearch.infrastructure.distribute import ResourceMetrics

        m = ResourceMetrics()
        assert m.cpu_percent == 0
        assert m.memory_percent == 0
        assert m.gpu_percent == 0
        assert m.timestamp is not None


class TestCostEstimate:
    """Tests for CostEstimate dataclass."""

    def test_defaults(self):
        from autoresearch.infrastructure.distribute import CloudProvider, CostEstimate

        est = CostEstimate(
            provider=CloudProvider.AWS,
            instance_type="p3.2xlarge",
            hourly_rate=3.06,
            estimated_hours=24,
            total_cost=73.44,
        )
        assert est.provider == CloudProvider.AWS
        assert est.total_cost == 73.44
        assert est.currency == "USD"


class TestNode:
    """Tests for the Node class."""

    def test_create(self):
        from autoresearch.infrastructure.distribute import Node, NodeConfig, NodeRole

        config = NodeConfig(role=NodeRole.WORKER, name="test-node")
        node = Node(config)
        assert node.id == "worker-test-node"
        assert node.status == "pending"

    def test_healthy_when_running(self):
        from autoresearch.infrastructure.distribute import Node, NodeConfig

        node = Node(NodeConfig())
        node.status = "running"
        node.metrics.cpu_percent = 50
        node.metrics.memory_percent = 60
        assert node.is_healthy() is True

    def test_unhealthy_high_cpu(self):
        from autoresearch.infrastructure.distribute import Node, NodeConfig

        node = Node(NodeConfig())
        node.status = "running"
        node.metrics.cpu_percent = 95
        assert node.is_healthy() is False

    def test_unhealthy_pending(self):
        from autoresearch.infrastructure.distribute import Node, NodeConfig

        node = Node(NodeConfig())
        assert node.is_healthy() is False

    def test_to_dict(self):
        from autoresearch.infrastructure.distribute import Node, NodeConfig

        node = Node(NodeConfig())
        d = node.to_dict()
        assert "id" in d
        assert "status" in d
        assert "config" in d
        assert "metrics" in d

    def test_experiments_completed(self):
        from autoresearch.infrastructure.distribute import Node, NodeConfig

        node = Node(NodeConfig())
        assert node.experiments_completed == 0
        node.experiments_completed = 5
        assert node.experiments_completed == 5
