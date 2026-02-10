# SMARTcgms – Personal CGM Intelligence Layer

Read-only, deterministic glucose alerting utilities designed for safe personal monitoring workflows.

## Features

- Strongly-typed glucose reading model (`GlucoseReading`).
- EWMA signal smoothing for noisy data.
- Conservative compression-low suppression heuristic.
- 30-minute predictive low warnings (linear projection).
- Sustained low detection.
- Alert cooldown to reduce alert fatigue.
- Medical-style decision logging with reasons.

## Safety posture

- No insulin dosing or automation logic.
- No sensor tampering features.
- Deterministic explainable rules only.
- Fail-safe defaults (`False` when signal quality is insufficient).

## Quick start

```python
from datetime import datetime
from cgm_intelligence import GlucoseReading, AlertDecisionEngine, DecisionLogger, process_new_reading

readings = [
    GlucoseReading(datetime.now(), glucose_mg_dl=85, trend_mg_dl_per_min=-0.7, motion=0.02),
]

engine = AlertDecisionEngine()
logger = DecisionLogger("cgm_decisions.log")

result = process_new_reading(readings, engine, logger, alert_callback=print)
print(result)
```

## Run tests

```bash
python -m unittest discover -s tests -v
```
