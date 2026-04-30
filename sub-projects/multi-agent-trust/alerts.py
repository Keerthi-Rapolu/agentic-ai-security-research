"""Structured alert emission for the unified security alert bus."""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import urllib.request
import urllib.error

logger = logging.getLogger(__name__)


@dataclass
class AlertPayload:
    """Structured alert payload matching the unified alert bus schema.

    Attributes:
        alert_id: Unique identifier for this alert (UUID string).
        sub_system: Source sub-system (``"rag_corpus"``, ``"alignment"``,
            ``"trust_propagation"``).
        severity: Alert severity (``"P1"``, ``"P2"``, ``"P3"``).
        timestamp: Unix timestamp of alert generation.
        title: Short human-readable alert title.
        description: Longer description of what was detected.
        evidence: Structured evidence dict (sub-system specific).
        recommended_action: Suggested remediation steps.
        metadata: Optional extra fields (pipeline ID, window ID, etc.).
    """

    alert_id: str
    sub_system: str
    severity: str
    timestamp: float
    title: str
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    recommended_action: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


def _new_alert_id() -> str:
    import uuid

    return str(uuid.uuid4())


class AlertEmitter:
    """Emits structured alert payloads to stdout and optionally a webhook.

    Alert payloads are serialised as newline-delimited JSON to stdout so they
    can be consumed by any log aggregator (Prometheus pushgateway, Datadog,
    Splunk, etc.).

    Args:
        webhook_url: Optional HTTPS endpoint for HTTP POST delivery.
        webhook_timeout: Request timeout in seconds for webhook calls.
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        webhook_timeout: float = 5.0,
    ) -> None:
        self.webhook_url = webhook_url
        self.webhook_timeout = webhook_timeout

    # ------------------------------------------------------------------
    # Sub-system specific emit methods
    # ------------------------------------------------------------------

    def emit_corpus_alert(
        self,
        severity: str,
        window_id: str,
        drift_score: float,
        threshold: float,
        candidate_count: int,
        top_candidates: Optional[List[Dict[str, Any]]] = None,
        **metadata: Any,
    ) -> AlertPayload:
        """Emit a RAG corpus poisoning drift alert.

        Args:
            severity: Alert severity (``"P1"``/``"P2"``/``"P3"``).
            window_id: The snapshot window that triggered the alert.
            drift_score: Observed JSD drift score.
            threshold: Configured detection threshold.
            candidate_count: Number of identified poisoning candidate documents.
            top_candidates: Optional list of top candidate doc dicts.
            **metadata: Additional metadata to include.

        Returns:
            The emitted :class:`AlertPayload`.
        """
        payload = AlertPayload(
            alert_id=_new_alert_id(),
            sub_system="rag_corpus",
            severity=severity,
            timestamp=time.time(),
            title=f"RAG Corpus Poisoning Detected — Window {window_id}",
            description=(
                f"Temporal JSD drift score {drift_score:.4f} exceeded threshold "
                f"{threshold:.4f} in snapshot window {window_id}. "
                f"{candidate_count} candidate poisoning document(s) identified."
            ),
            evidence={
                "window_id": window_id,
                "drift_score": drift_score,
                "threshold": threshold,
                "candidate_count": candidate_count,
                "top_candidates": top_candidates or [],
            },
            recommended_action=(
                "1. Review flagged documents in the provenance graph. "
                "2. Revoke ingestion credentials for suspicious sources. "
                "3. Quarantine candidate documents pending human review."
            ),
            metadata=metadata,
        )
        self._emit(payload)
        return payload

    def emit_alignment_alert(
        self,
        severity: str,
        mmd2: float,
        cusum_statistic: float,
        decision_threshold: float,
        regression_type: Optional[str] = None,
        step: Optional[int] = None,
        **metadata: Any,
    ) -> AlertPayload:
        """Emit an alignment regression detection alert.

        Args:
            severity: Alert severity.
            mmd2: Observed MMD² value that triggered the alert.
            cusum_statistic: CUSUM statistic value at alert time.
            decision_threshold: CUSUM decision threshold h.
            regression_type: ``"alignment"``, ``"capability"``, or ``"mixed"``.
            step: CUSUM step counter at alert time.
            **metadata: Additional metadata.

        Returns:
            The emitted :class:`AlertPayload`.
        """
        payload = AlertPayload(
            alert_id=_new_alert_id(),
            sub_system="alignment",
            severity=severity,
            timestamp=time.time(),
            title=f"Alignment Regression Detected ({severity})",
            description=(
                f"CUSUM statistic {cusum_statistic:.4f} exceeded threshold "
                f"{decision_threshold:.4f}. MMD²={mmd2:.6f}. "
                f"Regression type: {regression_type or 'unknown'}."
            ),
            evidence={
                "mmd2": mmd2,
                "cusum_statistic": cusum_statistic,
                "decision_threshold": decision_threshold,
                "regression_type": regression_type,
                "cusum_step": step,
            },
            recommended_action=(
                "1. Review sampled production outputs for constitutional violations. "
                "2. Inspect recent prompt distribution changes. "
                "3. Escalate to AI governance team if severity is P1."
            ),
            metadata=metadata,
        )
        self._emit(payload)
        return payload

    def emit_trust_alert(
        self,
        severity: str,
        agent_id: str,
        intrinsic_trust: float,
        effective_trust: float,
        tau_c: float,
        tau_eff_c: float,
        warning_tags: Optional[List[str]] = None,
        contaminated_descendants: Optional[List[str]] = None,
        **metadata: Any,
    ) -> AlertPayload:
        """Emit a trust propagation / Byzantine agent alert.

        Args:
            severity: Alert severity.
            agent_id: Agent that triggered the alert.
            intrinsic_trust: Agent's intrinsic trust score.
            effective_trust: Agent's effective trust score.
            tau_c: Byzantine threshold.
            tau_eff_c: Effective trust flagging threshold.
            warning_tags: List of warning tag strings from the interceptor.
            contaminated_descendants: Downstream agents with degraded trust.
            **metadata: Additional metadata.

        Returns:
            The emitted :class:`AlertPayload`.
        """
        payload = AlertPayload(
            alert_id=_new_alert_id(),
            sub_system="trust_propagation",
            severity=severity,
            timestamp=time.time(),
            title=f"Trust Violation — Agent {agent_id} ({severity})",
            description=(
                f"Agent {agent_id} intrinsic trust {intrinsic_trust:.3f} / "
                f"effective trust {effective_trust:.3f}. "
                f"τ_c={tau_c}, τ_eff_c={tau_eff_c}. "
                f"Tags: {warning_tags or []}."
            ),
            evidence={
                "agent_id": agent_id,
                "intrinsic_trust": intrinsic_trust,
                "effective_trust": effective_trust,
                "tau_c": tau_c,
                "tau_eff_c": tau_eff_c,
                "warning_tags": warning_tags or [],
                "contaminated_descendants": contaminated_descendants or [],
            },
            recommended_action=(
                "1. Inspect agent output history for anomalous behaviour. "
                "2. If Byzantine, consider quarantine and substitute with fallback agent. "
                "3. Recompute effective trust for all downstream agents."
            ),
            metadata=metadata,
        )
        self._emit(payload)
        return payload

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _emit(self, payload: AlertPayload) -> None:
        """Serialise payload to stdout and optionally POST to webhook."""
        line = json.dumps(asdict(payload), separators=(",", ":"))
        print(line, flush=True)
        logger.info(
            "Alert emitted: sub_system=%s severity=%s alert_id=%s",
            payload.sub_system, payload.severity, payload.alert_id,
        )

        if self.webhook_url:
            self._post_webhook(payload)

    def _post_webhook(self, payload: AlertPayload) -> None:
        body = json.dumps(asdict(payload)).encode("utf-8")
        req = urllib.request.Request(
            self.webhook_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.webhook_timeout) as resp:
                logger.debug("Webhook delivered: status=%d", resp.status)
        except urllib.error.URLError as exc:
            logger.error("Webhook delivery failed: %s", exc)
