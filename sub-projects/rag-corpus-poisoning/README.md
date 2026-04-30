# Sub-Project 1 — Temporal RAG Corpus Poisoning Detection

Detects gradual adversarial semantic drift in RAG knowledge bases by comparing embedding distributions across time windows.

## How it works

1. **Embedder** (`embedder.py`) — Embeds documents using `instructor-xl` (fallback: `all-mpnet-base-v2`), L2-normalised.
2. **SnapshotStore** (`snapshot_store.py`) — Persists versioned per-window snapshots in Qdrant; caches centroids in Redis.
3. **Detector** (`detector.py`) — Two-stage detection:
   - Stage 1: Fast centroid L2 distance check (`delta_fast = 0.05`).
   - Stage 2: Jensen-Shannon divergence via random projection to 32 dims.
4. **ProvenanceGraph** (`provenance.py`) — Counterfactual attribution of drift to specific documents; NetworkX graph of similar candidates.

## Configuration

All thresholds live in `config.py` → `DetectorConfig`:

| Parameter | Default | Meaning |
|---|---|---|
| `theta_corpus` | 0.15 | JSD alert threshold |
| `delta_fast` | 0.05 | Centroid fast-check threshold |
| `projection_dims` | 32 | Random projection dimensionality |
| `window_size` | 4 | Sliding window in snapshots |
| `hdbscan_min_cluster_size` | 3 | HDBSCAN minimum cluster size |
| `sigma_similarity` | 0.85 | Cosine similarity for provenance edges |

## Quick start

```bash
pip install -r requirements.txt

python - <<'EOF'
import numpy as np
from config import DetectorConfig
from detector import TemporalCorpusDriftDetector
from snapshot_store import Snapshot

cfg = DetectorConfig(projection_dims=8)
detector = TemporalCorpusDriftDetector(config=cfg)

rng = np.random.default_rng(0)
snap1 = Snapshot("W1", rng.standard_normal((100, 64)).astype("float32"))
# Poisoned snapshot — add 20 out-of-distribution docs
poison = rng.standard_normal((20, 64)).astype("float32") + 5.0
snap2 = Snapshot("W2", np.vstack([snap1.embeddings, poison]))

events = detector.detect([snap1, snap2])
print(f"Anomaly detected: {events[0].is_anomaly}  JSD={events[0].drift_score:.4f}")
EOF
```

## Run tests

```bash
pytest tests/ -v
```
