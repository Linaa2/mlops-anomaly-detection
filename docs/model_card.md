# Model Card — Hydraulic System Condition Monitor

## Model Details

| Field | Value |
|-------|-------|
| **Model type** | `MultiOutputClassifier(RandomForestClassifier)` |
| **Framework** | scikit-learn |
| **n_estimators** | 100 |
| **random_state** | 42 |
| **Training code** | `src/train.py` |
| **Registry** | MLflow Model Registry (`hydraulic-anomaly-detector`) |

## Intended Use

Predict the condition of 4 hydraulic system components from 17 sensor readings per operating cycle. Designed for condition monitoring in industrial settings to enable predictive maintenance.

**Not intended for**: safety-critical real-time control, other types of hydraulic systems, or extrapolation beyond the sensor ranges seen in training data.

## Training Data

**Source**: [UCI Condition Monitoring of Hydraulic Systems](https://archive.ics.uci.edu/ml/datasets/Condition+monitoring+of+hydraulic+systems)

- **Cycles**: ~2205 total, filtered to stable cycles only (`stable_flag == 0`)
- **Train/test split**: 80/20 (stratified on `cooler_condition`)
- **Features (17)**: PS1–PS6, EPS1, FS1–FS2, TS1–TS4, VS1, CE, CP, SE
- **Feature engineering**: Mean value per cycle per sensor (temporal aggregation from raw time-series)

## Targets (4 outputs)

| Target | Component | Classes | Description |
|--------|-----------|---------|-------------|
| `cooler_condition` | Cooler | 3, 20, 100 | Close to total failure → Full efficiency |
| `valve_condition` | Valve | 73, 80, 90, 100 | Optimal switching → Severe lag |
| `pump_leakage` | Pump | 0, 1, 2 | No leakage → Severe leakage |
| `accumulator_pressure` | Accumulator | 90, 100, 115, 130 | Close to total failure → Optimal pressure |

## Metrics

- **Primary**: F1 macro per target (`f1_macro_{target}`)
- **Aggregate**: Mean of per-target F1 macros (`f1_macro`)
- All metrics logged to MLflow per training run

## Continuous Training

The model is retrained daily via Airflow:

1. `data_pipeline` DAG samples 80% of the dataset (simulates new data)
2. `training_pipeline` DAG trains, logs to MLflow, registers in Model Registry
3. Champion/challenger comparison: new model promoted to Production only if `f1_macro` exceeds the current Production model

## Limitations

- **Static dataset**: The UCI dataset is fixed; random 80% sampling simulates data variability but does not reflect true distribution shift
- **No temporal features**: Only per-cycle means are used; within-cycle patterns (trends, spikes) are lost
- **Class imbalance**: Some targets have imbalanced classes (e.g., pump_leakage has mostly class 0)
- **No confidence calibration**: Raw class predictions, no probability calibration applied
- **Single aggregation**: Only mean per sensor per cycle — no std, min, max, percentiles

## Ethical Considerations

- No personally identifiable information (PII) in the dataset
- Model decisions should supplement, not replace, human expert judgment in maintenance planning
- False negatives (missed failures) are more costly than false positives in a maintenance context — recall should be monitored alongside F1
