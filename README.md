# Localized Time-Series Velocity Baseline Utilities

A clean, high-performance mathematical toolkit engineered to perform time-series decomposition, outlier mitigation, and variance profiling on sparse multi-index point-of-sale datasets.

## Core Features
* **Micro-Horizon Capacity Valuation:** Pinpoints localized time windows where system velocities experience standard deviations below historical trends.
* **Stochastic Outlier Control:** Robust preprocessing modules to isolate network latency drops from structural operational freezes.
* **Macro-Temporal Structural Decomposition:** Uses additive multi-layered time-series filters to parse trend, cycle, and structural day-of-week closures.

## Installation
```bash
pip install .


## Quick Start

You can use the baseline filter by passing a Pandas DataFrame containing your raw transaction logs. The DataFrame requires four columns: `zip_code`, `mcc`, `timestamp`, and `volume`.

```python
import pandas as pd
from micro import calculate_optimized_baseline

# 1. Create dummy transaction logs (e.g., a register processing transactions)
data = {
    "zip_code": ["90210"] * 5,
    "mcc": ["5411"] * 5,  # Grocery Stores
    "timestamp": [
        "2026-06-28 09:00:00",
        "2026-06-28 09:15:00",
        "2026-06-28 09:30:00",
        "2026-06-28 09:45:00",
        "2026-06-28 10:00:00"  # Sudden silent dropout
    ],
    "volume": [120, 115, 130, 125, 5]  # Massive sudden drop in transaction volume
}

df = pd.DataFrame(data)

# 2. Run the optimization baseline filter (alpha controls detection sensitivity)
dead_zones = calculate_optimized_baseline(df, alpha=0.25)

print(dead_zones)
