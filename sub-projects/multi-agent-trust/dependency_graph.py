"""Agent dependency graph backed by NetworkX."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentNode:
    """Metadata stored on each node of the dependency graph.

    Attributes:
        agent_id: Unique agent identifier.
        role: Agent role string used for trust prior lookup.
        initial_trust: Trust score assigned at registration.
        metadata: Arbitrary additional metadata.
    """

    agent_id: str
    role: str
    initial_trust: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentDependencyGraph:
    """NetworkX DiGraph representation of the agent dependency pipeline.

    Nodes represent agents; a directed edge ``(A → B)`` means agent B
    receives output from agent A.  Effective trust propagation follows the
    graph topologically.

    Args:
        None
    """

    def __init__(self) -> None:
        import networkx as nx

        self._graph: Any = nx.DiGraph()

    def add_agent(
        self,
        agent_id: str,
        role: str = "default",
        initial_trust: float = 0.60,
        **metadata: Any,
    ) -> None:
        """Register a new agent node in the dependency graph.

        Args:
            agent_id: Unique agent identifier.
            role: Agent role string (used for trust prior lookup).
            initial_trust: Starting trust score for this agent.
            **metadata: Arbitrary metadata stored on the node.
        """
        node = AgentNode(
            agent_id=agent_id,
            role=role,
            initial_trust=initial_trust,
            metadata=metadata,
        )
        self._graph.add_node(agent_id, data=node)
        logger.debug("Agent registered: %s (role=%s, tau_0=%.2f)", agent_id, role, initial_trust)

    def add_dependency(self, from_id: str, to_id: str) -> None:
        """Add a directed dependency edge: *to_id* depends on *from_id*.

        Args:
            from_id: Upstream agent ID (provider).
            to_id: Downstream agent ID (consumer).

        Raises:
            ValueError: If either agent ID is not in the graph.
        """
        if from_id not in self._graph:
            raise ValueError(f"Agent '{from_id}' not registered. Call add_agent first.")
        if to_id not in self._graph:
            raise ValueError(f"Agent '{to_id}' not registered. Call add_agent first.")
        self._graph.add_edge(from_id, to_id)
        logger.debug("Dependency added: %s → %s", from_id, to_id)

    def get_ancestors(self, agent_id: str) -> List[str]:
        """Return all ancestors of *agent_id* in topological order.

        Args:
            agent_id: Agent whose ancestors are needed.

        Returns:
            List of ancestor agent IDs.  Empty if the agent has no parents.
        """
        import networkx as nx

        if agent_id not in self._graph:
            return []
        return list(nx.ancestors(self._graph, agent_id))

    def get_parents(self, agent_id: str) -> List[str]:
        """Return the direct predecessors (parents) of *agent_id*.

        Args:
            agent_id: Agent ID.

        Returns:
            List of direct parent agent IDs.
        """
        if agent_id not in self._graph:
            return []
        return list(self._graph.predecessors(agent_id))

    def get_descendants(self, agent_id: str) -> List[str]:
        """Return all descendants of *agent_id*.

        Args:
            agent_id: Agent ID.

        Returns:
            List of descendant agent IDs.
        """
        import networkx as nx

        if agent_id not in self._graph:
            return []
        return list(nx.descendants(self._graph, agent_id))

    def get_trust_map(self) -> Dict[str, float]:
        """Return a dict mapping agent IDs to their ``initial_trust`` scores.

        Returns:
            Dict ``{agent_id: initial_trust}`` for all registered agents.
        """
        return {
            nid: data["data"].initial_trust
            for nid, data in self._graph.nodes(data=True)
            if "data" in data
        }

    def topological_sort(self) -> List[str]:
        """Return agent IDs in topological order (sources first).

        Returns:
            List of agent IDs in a valid topological ordering.

        Raises:
            ValueError: If the graph contains a cycle.
        """
        import networkx as nx

        try:
            return list(nx.topological_sort(self._graph))
        except nx.NetworkXUnfeasible as exc:
            raise ValueError(f"Dependency graph contains a cycle: {exc}") from exc

    def get_node(self, agent_id: str) -> Optional[AgentNode]:
        """Return the :class:`AgentNode` for *agent_id* or None.

        Args:
            agent_id: Agent ID.

        Returns:
            :class:`AgentNode` or ``None`` if not found.
        """
        node_data = self._graph.nodes.get(agent_id)
        if node_data is None:
            return None
        return node_data.get("data")

    def has_agent(self, agent_id: str) -> bool:
        """Return True if *agent_id* is registered.

        Args:
            agent_id: Agent ID.

        Returns:
            Boolean.
        """
        return agent_id in self._graph

    @property
    def agent_count(self) -> int:
        """Number of registered agents."""
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        """Number of dependency edges."""
        return self._graph.number_of_edges()
