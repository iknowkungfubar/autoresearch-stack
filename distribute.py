"""
Distribution - Multi-node execution and cloud deployment.

Phase 6.2: Distribution.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum


class NodeRole(Enum):
    """Node roles in distributed system."""

    MASTER = "master"
    WORKER = "worker"
    SCHEDULER = "scheduler"


class CloudProvider(Enum):
    """Supported cloud providers."""

    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    LOCAL = "local"


@dataclass
class NodeConfig:
    """Node configuration."""

    role: NodeRole = NodeRole.WORKER
    name: str = "worker-1"
    cpu_cores: int = 4
    memory_gb: int = 16
    gpu_count: int = 0
    gpu_type: Optional[str] = None
    region: str = "us-east-1"
    spot_instance: bool = False


@dataclass
class ResourceMetrics:
    """Resource usage metrics."""

    cpu_percent: float = 0
    memory_percent: float = 0
    gpu_percent: float = 0
    gpu_memory_percent: float = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CostEstimate:
    """Cost estimation."""

    provider: CloudProvider
    instance_type: str
    hourly_rate: float
    estimated_hours: float
    total_cost: float
    currency: str = "USD"


class Node:
    """Represents a node in the distributed system."""

    def __init__(self, config: NodeConfig):
        self.config = config
        self.status = "pending"
        self.id = f"{config.role.value}-{config.name}"
        self.metrics = ResourceMetrics()
        self.experiments_completed = 0

    def is_healthy(self) -> bool:
        """Check if node is healthy."""
        return (
            self.status == "running"
            and self.metrics.cpu_percent < 90
            and self.metrics.memory_percent < 90
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "status": self.status,
            "config": {
                "role": self.config.role.value,
                "name": self.config.name,
                "cpu_cores": self.config.cpu_cores,
                "memory_gb": self.config.memory_gb,
                "gpu_count": self.config.gpu_count,
            },
            "metrics": {
                "cpu_percent": self.metrics.cpu_percent,
                "memory_percent": self.metrics.memory_percent,
                "gpu_percent": self.metrics.gpu_percent,
            },
            "experiments_completed": self.experiments_completed,
        }


class Cluster:
    """Manages a distributed cluster of nodes."""

    def __init__(self, name: str = "autoresearch-cluster"):
        self.name = name
        self.nodes: Dict[str, Node] = {}
        self.master: Optional[Node] = None

    def add_node(self, config: NodeConfig) -> Node:
        """Add a node to the cluster."""
        node = Node(config)
        self.nodes[node.id] = node

        if config.role == NodeRole.MASTER:
            self.master = node

        return node

    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the cluster."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            return True
        return False

    def get_workers(self) -> List[Node]:
        """Get all worker nodes."""
        return [n for n in self.nodes.values() if n.config.role == NodeRole.WORKER]

    def get_healthy_workers(self) -> List[Node]:
        """Get healthy worker nodes."""
        return [n for n in self.get_workers() if n.is_healthy()]

    def get_least_loaded_worker(self) -> Optional[Node]:
        """Get the worker with lowest load."""
        workers = self.get_healthy_workers()
        if not workers:
            return None
        return min(workers, key=lambda w: w.metrics.cpu_percent)

    def total_resources(self) -> Dict[str, int]:
        """Calculate total cluster resources."""
        return {
            "cpu_cores": sum(n.config.cpu_cores for n in self.nodes.values()),
            "memory_gb": sum(n.config.memory_gb for n in self.nodes.values()),
            "gpu_count": sum(n.config.gpu_count for n in self.nodes.values()),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "total_resources": self.total_resources(),
        }


class ResourceManager:
    """Manages resources across the cluster."""

    def __init__(self, cluster: Cluster):
        self.cluster = cluster

    def allocate(
        self,
        required_cpu: int = 4,
        required_memory_gb: int = 16,
        required_gpus: int = 0,
    ) -> Optional[Node]:
        """Allocate a node that meets requirements."""
        for worker in self.cluster.get_healthy_workers():
            if (
                worker.config.cpu_cores >= required_cpu
                and worker.config.memory_gb >= required_memory_gb
                and worker.config.gpu_count >= required_gpus
            ):
                return worker
        return None

    def release(self, node: Node):
        """Release a node (mark as available)."""
        # In practice, would update node state
        pass

    def rebalance(self):
        """Rebalance workload across nodes."""
        # Placeholder for load balancing logic
        pass


class CostEstimator:
    """Estimates cloud deployment costs."""

    # Sample instance pricing (hourly)
    INSTANCE_RATES = {
        CloudProvider.AWS: {
            "t3.medium": 0.0416,
            "t3.large": 0.0832,
            "p3.2xlarge": 3.06,
            "p4d.24xlarge": 32.77,
        },
        CloudProvider.GCP: {
            "n1-standard-4": 0.19,
            "n1-standard-8": 0.38,
            "a2-highgpu-1g": 3.67,
        },
        CloudProvider.AZURE: {
            "Standard_D4s_v3": 0.192,
            "Standard_D8s_v3": 0.384,
            "NC24ads_A100_v4": 3.67,
        },
    }

    @classmethod
    def estimate(
        cls,
        provider: CloudProvider,
        instance_type: str,
        num_nodes: int,
        hours: float,
        spot_discount: float = 0.6,
    ) -> CostEstimate:
        """Estimate cost for a deployment."""
        rate = cls.INSTANCE_RATES.get(provider, {}).get(instance_type, 1.0)

        # Apply spot discount if applicable
        effective_rate = rate * (1 - spot_discount)

        total = effective_rate * num_nodes * hours

        return CostEstimate(
            provider=provider,
            instance_type=instance_type,
            hourly_rate=effective_rate,
            estimated_hours=hours,
            total_cost=total,
        )

    @classmethod
    def estimate_experiment_run(
        cls,
        provider: CloudProvider,
        instance_type: str,
        num_experiments: int,
        minutes_per_experiment: float,
        num_nodes: int = 1,
    ) -> CostEstimate:
        """Estimate cost for running experiments."""
        hours = (num_experiments * minutes_per_experiment) / 60
        return cls.estimate(provider, instance_type, num_nodes, hours)


class DistributedExecutor:
    """Executes experiments across the cluster."""

    def __init__(self, cluster: Cluster):
        self.cluster = cluster
        self.resource_manager = ResourceManager(cluster)
        self.running_tasks: Dict[str, Node] = {}

    def submit_task(
        self,
        task_id: str,
        task_config: Dict[str, Any],
    ) -> bool:
        """Submit a task to an available node."""
        node = self.resource_manager.allocate(
            required_cpu=task_config.get("cpu", 4),
            required_memory_gb=task_config.get("memory", 16),
            required_gpus=task_config.get("gpus", 0),
        )

        if node:
            self.running_tasks[task_id] = node
            node.status = "running"
            return True
        return False

    def get_task_status(self, task_id: str) -> Optional[str]:
        """Get status of a task."""
        if task_id in self.running_tasks:
            return self.running_tasks[task_id].status
        return None

    def complete_task(self, task_id: str):
        """Mark a task as complete."""
        if task_id in self.running_tasks:
            node = self.running_tasks[task_id]
            node.experiments_completed += 1
            node.status = "idle"
            del self.running_tasks[task_id]
            self.resource_manager.release(node)


# Kubernetes deployment helpers
def generate_k8s_deployment(
    name: str = "autoresearch",
    replicas: int = 3,
    cpu_request: str = "2",
    memory_request: str = "4Gi",
    gpu_count: int = 0,
) -> str:
    """Generate Kubernetes deployment YAML."""
    gpu_resource = ""
    if gpu_count > 0:
        gpu_resource = f"""
        nvidia.com/gpu: {gpu_count}"""

    return f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: {name}
  template:
    metadata:
      labels:
        app: {name}
    spec:
      containers:
      - name: {name}
        image: autoresearch:latest
        resources:
          requests:
            cpu: {cpu_request}
            memory: {memory_request}{gpu_resource}
          limits:
            cpu: {cpu_request}
            memory: {memory_request}{gpu_resource}
"""


def generate_k8s_service(name: str = "autoresearch") -> str:
    """Generate Kubernetes service YAML."""
    return f"""apiVersion: v1
kind: Service
metadata:
  name: {name}
spec:
  selector:
    app: {name}
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
"""


def generate_docker_compose(num_workers: int = 3) -> str:
    """Generate Docker Compose for local cluster."""
    services = {
        "master": {
            "image": "autoresearch:latest",
            "ports": ["8000:8000"],
            "environment": ["ROLE=master"],
        }
    }

    for i in range(num_workers):
        services[f"worker-{i + 1}"] = {
            "image": "autoresearch:latest",
            "environment": [f"ROLE=worker", "MASTER_HOST=master"],
        }

    return json.dumps({"services": services}, indent=2)


if __name__ == "__main__":
    print("Testing distribution module...")

    # Create cluster
    cluster = Cluster("test-cluster")

    # Add master
    cluster.add_node(
        NodeConfig(role=NodeRole.MASTER, name="master", cpu_cores=8, memory_gb=32)
    )

    # Add workers
    for i in range(3):
        cluster.add_node(
            NodeConfig(
                role=NodeRole.WORKER,
                name=f"worker-{i + 1}",
                cpu_cores=8,
                memory_gb=32,
                gpu_count=1,
            )
        )

    print(f"Cluster: {cluster.name}")
    print(f"Nodes: {len(cluster.nodes)}")
    print(f"Total resources: {cluster.total_resources()}")

    # Test cost estimation
    cost = CostEstimator.estimate_experiment_run(
        provider=CloudProvider.AWS,
        instance_type="p3.2xlarge",
        num_experiments=100,
        minutes_per_experiment=30,
        num_nodes=2,
    )
    print(f"Estimated cost: ${cost.total_cost:.2f}")

    # Generate K8s deployment
    k8s_yaml = generate_k8s_deployment()
    print("\nK8s Deployment:\n" + k8s_yaml[:500] + "...")
