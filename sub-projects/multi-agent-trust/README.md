# Sub-Project 3 — Multi-Agent Trust Propagation

Implements a formal trust propagation algebra for LLM agent pipelines with Byzantine fault detection and containment.

## How it works

1. **TrustRegistry** (`registry.py`) — Redis-backed hot tier + in-memory cold tier with SHA-256 hash-chained history.
2. **AgentDependencyGraph** (`dependency_graph.py`) — NetworkX DiGraph of agent dependencies with topological ordering.
3. **TrustEngine** (`trust_engine.py`) — Computes effective trust via `τ_eff = τ_j × ∏(τ_i^α)`, applies degradation/recovery, detects Byzantine agents.
4. **MessageInterceptor** (`interceptor.py`) — Intercepts inter-agent messages, attaches trust envelopes, flags low-trust senders.
5. **AlertEmitter** (`alerts.py`) — Emits structured JSON alert payloads to stdout and optional webhook.

## Trust propagation formula

```
τ_j_eff = τ_j × ∏(τ_i_eff ^ α)  for all parents i of j
```

An agent is **Byzantine** if `τ < τ_c` (default 0.30).  Output is **flagged** if `τ_eff < τ_eff_c` (default 0.25).

Trust recovers exponentially toward role prior with 24-hour half-life.

## Quick start

```bash
pip install -r requirements.txt

python - <<'EOF'
from config import TrustConfig
from dependency_graph import AgentDependencyGraph
from interceptor import AgentMessage, MessageInterceptor
from registry import TrustRegistry
from trust_engine import TrustEngine

config = TrustConfig()
registry = TrustRegistry()
graph = AgentDependencyGraph()
engine = TrustEngine(registry=registry, graph=graph, config=config)

engine.register_agent("reader", role="retrieval")
engine.register_agent("classifier", role="user_facing")
engine.register_agent("reporter", role="user_facing")
graph.add_dependency("reader", "classifier")
graph.add_dependency("classifier", "reporter")

# Simulate compromise
engine.quarantine("reader")
print("reporter effective trust:", engine.compute_effective_trust("reporter"))
print("Byzantine:", engine.is_byzantine("reader"))
EOF
```

## Run tests

```bash
pytest tests/ -v
```
