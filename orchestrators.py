"""
Agentic Orchestrator Integrations.

Supports:
- OpenCode
- OpenCrew
- AgentForge
- CrewAI
- AutoGen
- LangChain
- LlamaIndex

Phase 7.1: Multi-Orchestrator Support.
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum


# Tool type enum
class OrchestratorType(Enum):
    """Orchestrator types."""

    OPENCREW = "opencrew"
    AGENTFORGE = "agentforge"
    CREWAI = "crewai"
    AUTOGEN = "autogen"
    LANGCHAIN = "langchain"
    LLAMA_INDEX = "llama_index"
    NAVIGATOR = "navigator"


@dataclass
class AgentTask:
    """A task for an agent."""

    description: str
    expected_output: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class AgentResult:
    """Result from an agent."""

    content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseOrchestrator(ABC):
    """Abstract base class for orchestrators."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def run(self, task: AgentTask) -> AgentResult:
        """Run a task."""
        pass

    @abstractmethod
    def run_multi(self, tasks: List[AgentTask]) -> List[AgentResult]:
        """Run multiple tasks."""
        pass


class OpenCrewIntegrator(BaseOrchestrator):
    """OpenCrew (opencrew.ai) integration."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._client = None

    def _get_client(self):
        """Lazy load OpenCrew client."""
        if self._client is None:
            try:
                import opencrew

                self._client = opencrew
            except ImportError:
                self.logger.warning("opencrew not installed: pip install opencrew")
        return self._client

    def run(self, task: AgentTask) -> AgentResult:
        """Run a task with OpenCrew."""
        client = self._get_client()
        if not client:
            return AgentResult(
                content="OpenCrew not available",
                metadata={"error": "package_not_installed"},
            )

        try:
            # OpenCrew API call
            result = client.run(
                task=task.description,
                context=task.context or {},
            )
            return AgentResult(
                content=result.get("output", ""),
                tool_calls=result.get("tool_calls", []),
                metadata=result,
            )
        except Exception as e:
            return AgentResult(
                content=f"Error: {e}",
                metadata={"error": str(e)},
            )

    def run_multi(self, tasks: List[AgentTask]) -> List[AgentResult]:
        """Run multiple tasks."""
        return [self.run(task) for task in tasks]


class AgentForgeIntegrator(BaseOrchestrator):
    """AgentForge integration."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._client = None

    def _get_client(self):
        """Lazy load AgentForge client."""
        if self._client is None:
            try:
                import agentforge

                self._client = agentforge
            except ImportError:
                self.logger.warning("agentforge not installed: pip install agentforge")
        return self._client

    def run(self, task: AgentTask) -> AgentResult:
        """Run a task with AgentForge."""
        client = self._get_client()
        if not client:
            return AgentResult(
                content="AgentForge not available",
                metadata={"error": "package_not_installed"},
            )

        try:
            result = client.run_agent(
                task=task.description,
                context=task.context or {},
            )
            return AgentResult(
                content=str(result),
                metadata={"result": result},
            )
        except Exception as e:
            return AgentResult(
                content=f"Error: {e}",
                metadata={"error": str(e)},
            )

    def run_multi(self, tasks: List[AgentTask]) -> List[AgentResult]:
        """Run multiple tasks."""
        return [self.run(task) for task in tasks]


class CrewAIIntegrator(BaseOrchestrator):
    """CrewAI integration."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.crew = None

    def _setup_crew(self):
        """Set up CrewAI crew."""
        try:
            from crewai import Agent, Crew, Task, Process
            from langchain_openai import ChatOpenAI

            # Get LLM
            llm = self.config.get("llm", "gpt-4o")
            api_key = os.getenv("OPENAI_API_KEY")

            # Create agents
            researcher = Agent(
                role="Researcher",
                goal="Find and analyze research papers",
                backstory="Expert researcher",
                verbose=True,
            )

            # Create tasks
            tasks = []
            for task in self.config.get("tasks", []):
                tasks.append(
                    Task(
                        description=task,
                        agent=researcher,
                    )
                )

            # Create crew
            self.crew = Crew(
                agents=[researcher],
                tasks=tasks,
                process=Process.sequential,
                verbose=True,
            )

        except ImportError:
            self.logger.warning("crewai not installed: pip install crewai")

    def run(self, task: AgentTask) -> AgentResult:
        """Run a task with CrewAI."""
        if not self.crew:
            self._setup_crew()

        if not self.crew:
            return AgentResult(
                content="CrewAI not available",
                metadata={"error": "package_not_installed"},
            )

        try:
            result = self.crew.kickoff(inputs={"task": task.description})
            return AgentResult(
                content=str(result),
                metadata={"result": result},
            )
        except Exception as e:
            return AgentResult(
                content=f"Error: {e}",
                metadata={"error": str(e)},
            )

    def run_multi(self, tasks: List[AgentTask]) -> List[AgentResult]:
        """Run multiple tasks."""
        return [self.run(task) for task in tasks]


class AutoGenIntegrator(BaseOrchestrator):
    """AutoGen (microsoft/autogen) integration."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._client = None

    def _get_client(self):
        """Lazy load AutoGen client."""
        if self._client is None:
            try:
                import autogen

                self._client = autogen
            except ImportError:
                self.logger.warning("pyautogen not installed: pip install pyautogen")
        return self._client

    def run(self, task: AgentTask) -> AgentResult:
        """Run a task with AutoGen."""
        client = self._get_client()
        if not client:
            return AgentResult(
                content="AutoGen not available",
                metadata={"error": "package_not_installed"},
            )

        try:
            from autogen import ConversableAgent

            # Create assistant agent
            assistant = ConversableAgent(
                name="assistant",
                llm_config={
                    "model": self.config.get("model", "gpt-4o"),
                },
            )

            # Run conversation
            result = assistant.generate_reply(
                messages=[{"content": task.description, "role": "user"}]
            )

            return AgentResult(
                content=str(result),
                metadata={"result": result},
            )
        except Exception as e:
            return AgentResult(
                content=f"Error: {e}",
                metadata={"error": str(e)},
            )

    def run_multi(self, tasks: List[AgentTask]) -> List[AgentResult]:
        """Run multiple tasks."""
        return [self.run(task) for task in tasks]


class LangChainIntegrator(BaseOrchestrator):
    """LangChain integration."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._client = None

    def _get_client(self):
        """Lazy load LangChain client."""
        if self._client is None:
            try:
                from langchain.chat_models import ChatOpenAI
                from langchain.schema import HumanMessage

                self._client = {
                    "chat": ChatOpenAI(
                        model=self.config.get("model", "gpt-4o"),
                        temperature=self.config.get("temperature", 0.7),
                    ),
                }
            except ImportError:
                self.logger.warning("langchain not installed: pip install langchain")
        return self._client

    def run(self, task: AgentTask) -> AgentResult:
        """Run a task with LangChain."""
        client = self._get_client()
        if not client:
            return AgentResult(
                content="LangChain not available",
                metadata={"error": "package_not_installed"},
            )

        try:
            from langchain.schema import HumanMessage

            messages = [HumanMessage(content=task.description)]
            result = client["chat"].invoke(messages)

            return AgentResult(
                content=result.content,
                metadata={"response": result},
            )
        except Exception as e:
            return AgentResult(
                content=f"Error: {e}",
                metadata={"error": str(e)},
            )

    def run_multi(self, tasks: List[AgentTask]) -> List[AgentResult]:
        """Run multiple tasks."""
        return [self.run(task) for task in tasks]


class LlamaIndexIntegrator(BaseOrchestrator):
    """LlamaIndex integration."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._client = None

    def _get_client(self):
        """Lazy load LlamaIndex client."""
        if self._client is None:
            try:
                from llama_index import VectorStoreIndex, ServiceContext
                from llama_index.llms import ChatMessage

                self._client = {
                    "index": None,  # Set after loading data
                    "service_context": ServiceContext.from_defaults(),
                }
            except ImportError:
                self.logger.warning(
                    "llama-index not installed: pip install llama-index"
                )
        return self._client

    def run(self, task: AgentTask) -> AgentResult:
        """Run a task with LlamaIndex."""
        client = self._get_client()
        if not client:
            return AgentResult(
                content="LlamaIndex not available",
                metadata={"error": "package_not_installed"},
            )

        try:
            from llama_index import VectorStoreIndex

            # Query index
            index = self.config.get("index")
            if not index:
                return AgentResult(
                    content="No index configured",
                    metadata={"error": "no_index"},
                )

            result = index.as_query_engine().query(task.description)
            return AgentResult(
                content=str(result),
                metadata={"response": result},
            )
        except Exception as e:
            return AgentResult(
                content=f"Error: {e}",
                metadata={"error": str(e)},
            )

    def run_multi(self, tasks: List[AgentTask]) -> List[AgentResult]:
        """Run multiple tasks."""
        return [self.run(task) for task in tasks]


# Orchestrator factory
class OrchestratorFactory:
    """Factory for creating orchestrators."""

    ORCHESTRATORS = {
        OrchestratorType.OPENCREW: OpenCrewIntegrator,
        OrchestratorType.AGENTFORGE: AgentForgeIntegrator,
        OrchestratorType.CREWAI: CrewAIIntegrator,
        OrchestratorType.AUTOGEN: AutoGenIntegrator,
        OrchestratorType.LANGCHAIN: LangChainIntegrator,
        OrchestratorType.LLAMA_INDEX: LlamaIndexIntegrator,
    }

    @classmethod
    def create(
        cls,
        orchestrator: Union[OrchestratorType, str],
        config: Optional[Dict[str, Any]] = None,
    ) -> BaseOrchestrator:
        """Create an orchestrator instance."""
        if isinstance(orchestrator, str):
            orchestrator = OrchestratorType(orchestrator)

        orch_class = cls.ORCHESTRATORS.get(orchestrator)
        if not orch_class:
            raise ValueError(f"Unknown orchestrator: {orchestrator}")

        return orch_class(config=config)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> BaseOrchestrator:
        """Create orchestrator from config dict."""
        return cls.create(
            orchestrator=config.get("orchestrator", "langchain"),
            config=config.get("config", {}),
        )


# Unified orchestrator client
class OrchestratorClient:
    """Unified client for all orchestrators."""

    def __init__(self, orchestrator: Optional[BaseOrchestrator] = None):
        self.orchestrator = orchestrator
        self.logger = logging.getLogger("OrchestratorClient")

    def run(self, task: Union[str, AgentTask]) -> AgentResult:
        """Run a task."""
        if isinstance(task, str):
            task = AgentTask(description=task)
        if not self.orchestrator:
            raise RuntimeError("No orchestrator configured")
        return self.orchestrator.run(task)

    def run_multi(self, tasks: List[Union[str, AgentTask]]) -> List[AgentResult]:
        """Run multiple tasks."""
        parsed_tasks = [
            AgentTask(description=t) if isinstance(t, str) else t for t in tasks
        ]
        if not self.orchestrator:
            raise RuntimeError("No orchestrator configured")
        return self.orchestrator.run_multi(parsed_tasks)


if __name__ == "__main__":
    print("=== Testing Orchestrator Factory ===\n")

    # Test LangChain (default available)
    try:
        langchain = OrchestratorFactory.create(OrchestratorType.LANGCHAIN)
        print(f"Created: {langchain.__class__.__name__}")

        result = langchain.run(
            AgentTask(
                description="What is 2 + 2?",
            )
        )
        print(f"Result: {result.content}")

    except Exception as e:
        print(f"Error: {e}")

    print("\nSupported orchestrators:")
    for t in OrchestratorType:
        print(f"  - {t.value}")
