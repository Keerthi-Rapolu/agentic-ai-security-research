# Sub-Project 2 — Post-Deployment LLM Alignment Regression Monitor

Continuously monitors deployed LLM behavioral distributions against a constitutional baseline using Maximum Mean Discrepancy and CUSUM change-point detection.

## How it works

1. **ConstitutionalBaseline** (`baseline.py`) — Stores curated aligned responses as versioned, SHA-256-signed embedding snapshots.
2. **OutputSampler** (`sampler.py`) — Stratified sampling: 1% random, 5% principle-triggered, 100% safety-triggered.
3. **MMDComputer** (`mmd.py`) — Unbiased U-statistic MMD² estimator with RBF kernel and median heuristic bandwidth.
4. **CUSUMDetector** (`changepoint.py`) — Cumulative Sum change-point detection on the MMD² time series.
5. **AlignmentRegressionMonitor** (`monitor.py`) — Orchestrates the full pipeline; classifies drift as `capability`, `alignment`, or `mixed`.

## Configuration

All settings in `config.py` → `MonitorConfig`:

| Parameter | Default | Meaning |
|---|---|---|
| `theta_align` | 0.05 | MMD² alert threshold |
| `theta_max` | 0.30 | MMD² P1 severity ceiling |
| `cusum_decision_threshold` | 5.0 | CUSUM statistic h |
| `random_sample_rate` | 0.01 | 1% random baseline sampling |
| `principle_sample_rate` | 0.05 | 5% principle-triggered sampling |

## Quick start

```bash
pip install -r requirements.txt

python - <<'EOF'
from baseline import ConstitutionalBaseline
from monitor import AlignmentRegressionMonitor
from config import MonitorConfig

baseline = ConstitutionalBaseline()
baseline.fit(["I cannot help with that request.", "Here is accurate information about..."])

monitor = AlignmentRegressionMonitor(baseline=baseline, window_size=50)
for _ in range(200):
    result = monitor.ingest_output("some production output")
    if result.cusum_result and result.cusum_result.is_alert:
        print("ALERT:", result.cusum_result.severity)
        break

print("Status:", monitor.get_status())
EOF
```

## Run tests

```bash
pytest tests/ -v
```
