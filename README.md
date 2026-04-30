# Unified Security & Trust Infrastructure for Production Agentic AI Systems

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![arXiv](https://img.shields.io/badge/arXiv-placeholder-red)

> *From retrieval to reasoning to multi-agent orchestration — securing every layer of the production AI stack.*

**Author:** Keerthi Rapolu | **Status:** Research Design v1.0 | **Date:** April 2026

---

## Overview

This repository implements production-quality detection and monitoring infrastructure for three critical attack surfaces in deployed agentic AI systems that existing tools (LlamaGuard, Prompt Shield, UEBA, DLP) leave undefended:

| Sub-Project | Attack Surface | Detection Method |
|---|---|---|
| [RAG Corpus Poisoning Detection](#sub-project-1--rag-corpus-poisoning-detection) | Gradual embedding-level drift in RAG knowledge bases | Temporal JSD + HDBSCAN + provenance graph |
| [Alignment Regression Monitor](#sub-project-2--alignment-regression-monitor) | Silent behavioral drift from constitutional baseline | MMD² + CUSUM change-point detection |
| [Multi-Agent Trust Propagation](#sub-project-3--multi-agent-trust-propagation) | Byzantine agent contamination in LLM pipelines | Formal trust algebra + dependency graph |

For the full system architecture, see [DESIGN_DOC.md → Section 6](DESIGN_DOC.md#6-system-architecture-high-level).

---

## Sub-Project 1 — RAG Corpus Poisoning Detection

[`sub-projects/rag-corpus-poisoning/`](sub-projects/rag-corpus-poisoning/)

Detects gradual adversarial document injection into RAG corpora by comparing embedding-space distributions across time-windowed snapshots. When Jensen-Shannon divergence exceeds threshold, a provenance graph traces drift back to specific documents.

**Key files:** `detector.py`, `embedder.py`, `snapshot_store.py`, `provenance.py`, `config.py`

```bash
cd sub-projects/rag-corpus-poisoning
pip install -r requirements.txt
pytest tests/ -v
```

---

## Sub-Project 2 — Alignment Regression Monitor

[`sub-projects/alignment-regression-monitor/`](sub-projects/alignment-regression-monitor/)

Continuously monitors deployed LLM outputs against a signed constitutional baseline using Maximum Mean Discrepancy and CUSUM change-point detection. Classifies detected drift as capability regression, alignment regression, or mixed.

**Key files:** `monitor.py`, `mmd.py`, `changepoint.py`, `baseline.py`, `sampler.py`, `config.py`

```bash
cd sub-projects/alignment-regression-monitor
pip install -r requirements.txt
pytest tests/ -v
```

---

## Sub-Project 3 — Multi-Agent Trust Propagation

[`sub-projects/multi-agent-trust/`](sub-projects/multi-agent-trust/)

Implements a formal trust propagation algebra for LLM agent dependency graphs. Assigns, degrades, and propagates trust scores; detects Byzantine agents and flags contaminated downstream outputs with provable containment properties.

**Key files:** `trust_engine.py`, `registry.py`, `dependency_graph.py`, `interceptor.py`, `alerts.py`, `config.py`

```bash
cd sub-projects/multi-agent-trust
pip install -r requirements.txt
pytest tests/ -v
```

---

## Architecture

See [DESIGN_DOC.md → Section 6](DESIGN_DOC.md#6-system-architecture-high-level) for the full Mermaid architecture diagram showing how the three sub-systems connect through shared Qdrant, Redis, and alert bus infrastructure.

---

## Run all tests

```bash
pip install -r requirements.txt
pytest sub-projects/ -v
```

---

## References

- Greshake et al. (2023) — Indirect Prompt Injection in RAG systems
- Campello et al. (2013) — HDBSCAN: Hierarchical DBSCAN
- Page (1954) — CUSUM sequential analysis
- Lamport et al. (1982) — Byzantine Generals Problem
- Constitutional AI (Anthropic, 2022)

See [DESIGN_DOC.md → Section 20](DESIGN_DOC.md#20-references) for the full bibliography.
