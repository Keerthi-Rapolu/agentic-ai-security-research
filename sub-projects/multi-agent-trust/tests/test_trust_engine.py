"""Tests for the multi-agent trust propagation engine."""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from config import TrustConfig
from dependency_graph import AgentDependencyGraph
from interceptor import AgentMessage, MessageInterceptor
from registry import TrustRegistry
from trust_engine import TrustEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(tau_c=0.30, tau_eff_c=0.25, alpha=1.0):
    config = TrustConfig(tau_c=tau_c, tau_eff_c=tau_eff_c, alpha=alpha, half_life_hours=24.0)
    registry = TrustRegistry()  # in-memory only (no Redis required)
    graph = AgentDependencyGraph()
    return TrustEngine(registry=registry, graph=graph, config=config), config


def _linear_pipeline(engine, agents, trusts):
    """Register agents with given trust scores and wire them in a chain."""
    for aid, tau in zip(agents, trusts):
        engine.register_agent(aid, role="default", initial_trust=tau)
    for i in range(len(agents) - 1):
        engine.graph.add_dependency(agents[i], agents[i + 1])


# ---------------------------------------------------------------------------
# Test: trust propagation through linear pipeline
# ---------------------------------------------------------------------------

class TestTrustPropagation:
    """Effective trust propagates multiplicatively through a chain."""

    def test_linear_pipeline_effective_trust(self):
        """τ_eff for end agent = product of all intrinsic scores."""
        engine, config = _make_engine(alpha=1.0)
        agents = ["A", "B", "C"]
        trusts = [0.9, 0.8, 0.7]
        _linear_pipeline(engine, agents, trusts)

        eff_C = engine.compute_effective_trust("C")
        expected = 0.7 * 0.8 * 0.9
        assert abs(eff_C - expected) < 1e-5, f"Expected {expected:.4f}, got {eff_C:.4f}"

    def test_single_agent_eff_equals_intrinsic(self):
        """An isolated agent's effective trust equals its intrinsic score."""
        engine, _ = _make_engine()
        engine.register_agent("solo", initial_trust=0.75)
        assert abs(engine.compute_effective_trust("solo") - 0.75) < 1e-6

    def test_alpha_controls_decay_rate(self):
        """Higher alpha accelerates trust decay through the pipeline."""
        engine_1, _ = _make_engine(alpha=1.0)
        engine_2, _ = _make_engine(alpha=2.0)

        for eng in (engine_1, engine_2):
            _linear_pipeline(eng, ["X", "Y"], [0.8, 0.9])

        eff_1 = engine_1.compute_effective_trust("Y")  # 0.9 * 0.8^1 = 0.72
        eff_2 = engine_2.compute_effective_trust("Y")  # 0.9 * 0.8^2 = 0.576
        assert eff_1 > eff_2, "Higher alpha should yield lower effective trust"


# ---------------------------------------------------------------------------
# Test: Byzantine detection when tau < 0.3
# ---------------------------------------------------------------------------

class TestByzantineDetection:
    """Agents with intrinsic trust below tau_c are correctly classified."""

    def test_below_tau_c_is_byzantine(self):
        """Agents with trust < 0.3 are detected as Byzantine."""
        engine, config = _make_engine(tau_c=0.30)
        engine.register_agent("bad_actor", initial_trust=0.25)
        assert engine.is_byzantine("bad_actor") is True

    def test_above_tau_c_not_byzantine(self):
        """Agents with trust >= 0.3 are not Byzantine."""
        engine, _ = _make_engine(tau_c=0.30)
        engine.register_agent("honest", initial_trust=0.80)
        assert engine.is_byzantine("honest") is False

    def test_degradation_to_byzantine(self):
        """Sequential penalties should eventually classify agent as Byzantine."""
        engine, _ = _make_engine(tau_c=0.30)
        engine.register_agent("target", initial_trust=0.75)
        for _ in range(4):
            engine.apply_alignment_penalty("target")
        assert engine.is_byzantine("target"), "Agent should be Byzantine after repeated penalties"

    def test_quarantine_sets_trust_to_zero(self):
        """Quarantine immediately sets trust to effectively 0 (below tau_c)."""
        engine, _ = _make_engine()
        engine.register_agent("suspect", initial_trust=0.60)
        engine.quarantine("suspect")
        # Registry stores 0.0; get_trust may return a tiny positive value due to
        # floating-point exponential recovery over microseconds — assert < tau_c.
        assert engine.get_trust("suspect") < engine.config.tau_c
        assert engine.is_byzantine("suspect") is True


# ---------------------------------------------------------------------------
# Test: downstream contamination flags
# ---------------------------------------------------------------------------

class TestDownstreamContamination:
    """Downstream agents are flagged when a Byzantine ancestor is present."""

    def test_contaminated_descendants_identified(self):
        """All agents downstream of a Byzantine node have reduced effective trust."""
        engine, _ = _make_engine(tau_c=0.30, tau_eff_c=0.25)
        agents = ["Root", "Mid", "Leaf"]
        # Root is Byzantine (low trust)
        engine.register_agent("Root", initial_trust=0.10)
        engine.register_agent("Mid", initial_trust=0.90)
        engine.register_agent("Leaf", initial_trust=0.90)
        engine.graph.add_dependency("Root", "Mid")
        engine.graph.add_dependency("Mid", "Leaf")

        contaminated = engine.get_contaminated_descendants("Root")
        assert "Mid" in contaminated
        assert "Leaf" in contaminated

    def test_interceptor_flags_message_from_low_trust_sender(self):
        """MessageInterceptor should flag messages from agents below tau_eff_c."""
        engine, config = _make_engine(tau_c=0.30, tau_eff_c=0.25)
        engine.register_agent("compromised", initial_trust=0.10)
        engine.register_agent("receiver", initial_trust=0.90)

        interceptor = MessageInterceptor(engine=engine, config=config)
        msg = AgentMessage(
            sender_id="compromised",
            receiver_id="receiver",
            content="malicious payload",
        )
        result = interceptor.intercept(msg)
        assert result.is_flagged is True
        assert result.should_quarantine is True

    def test_interceptor_passes_high_trust_message(self):
        """Messages from high-trust agents should not be flagged."""
        engine, config = _make_engine()
        engine.register_agent("trusted_agent", initial_trust=0.95)
        engine.register_agent("dest", initial_trust=0.90)

        interceptor = MessageInterceptor(engine=engine, config=config)
        msg = AgentMessage(
            sender_id="trusted_agent",
            receiver_id="dest",
            content="normal message",
        )
        result = interceptor.intercept(msg)
        assert result.is_flagged is False
        assert result.should_quarantine is False


# ---------------------------------------------------------------------------
# Test: exponential recovery over time steps
# ---------------------------------------------------------------------------

class TestExponentialRecovery:
    """Trust recovers toward role prior after degradation."""

    def test_recovery_after_penalty(self):
        """After a penalty, trust should recover toward the initial prior over time."""
        engine, _ = _make_engine()
        engine.register_agent("recovering", role="user_facing", initial_trust=0.75)

        # Apply a penalty
        engine.apply_alignment_penalty("recovering")
        degraded = engine.get_trust("recovering")

        # Manually rewind the last event time to simulate elapsed time
        past_event = time.time() - 3600 * 48  # 48 hours ago
        engine._last_event_time["recovering"] = past_event

        recovered = engine.get_trust("recovering")
        assert recovered > degraded, (
            f"Trust should recover over time: degraded={degraded:.3f} recovered={recovered:.3f}"
        )

    def test_recovery_does_not_exceed_prior(self):
        """Trust should not overshoot above the role prior during recovery."""
        engine, _ = _make_engine()
        engine.register_agent("agent", role="default", initial_trust=0.60)
        engine.apply_alignment_penalty("agent")

        # Simulate a very long time elapsed (trust should converge to prior ~0.60)
        engine._last_event_time["agent"] = time.time() - 3600 * 1000
        recovered = engine.get_trust("agent")
        prior = engine._role_priors["agent"]
        assert recovered <= prior + 1e-4, f"Recovery should not exceed prior {prior:.3f}"


# ---------------------------------------------------------------------------
# Test: trust registry history and hash chaining
# ---------------------------------------------------------------------------

class TestTrustRegistry:
    """Registry persists history with correct SHA-256 hash chaining."""

    def test_history_appended_on_each_update(self):
        """Each set() call should append one entry to the history."""
        registry = TrustRegistry()
        for i in range(3):
            registry.set("agent_x", 0.8 - i * 0.1, reason=f"step {i}")
        history = registry.history("agent_x")
        assert len(history) == 3

    def test_history_hash_chaining(self):
        """Each history entry's prev_hash matches the previous entry's hash."""
        registry = TrustRegistry()
        registry.set("agent_y", 0.9, reason="init")
        registry.set("agent_y", 0.7, reason="penalty")
        registry.set("agent_y", 0.5, reason="second_penalty")

        history = registry.history("agent_y")
        assert history[1].prev_hash == history[0].entry_hash
        assert history[2].prev_hash == history[1].entry_hash
