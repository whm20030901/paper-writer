# TSK-PINN V1 Experiment Skeleton

## Objective
Build the first minimal, reproducible benchmark for a TSK-based local reliability weighted PINN.

## Recommended first benchmark
- PDE: 1D Burgers equation
- Baseline: vanilla PINN
- Proposed: PINN backbone + Gaussian TSK antecedent + first-order TSK consequent for local loss weights

## Suggested directory layout to ask Codex to create
```text
experiments/tsk_pinn_v1/
  configs/
    burgers_baseline.yaml
    burgers_tsk.yaml
  scripts/
    train_baseline.py
    train_tsk.py
    eval_compare.py
  src/
    data/
    models/
      backbone.py
      tsk.py
      losses.py
    trainers/
    utils/
  outputs/
    burgers_baseline/
    burgers_tsk/
  reports/
    burgers_v1_report.md
```

## Minimum success criteria
- both models run end-to-end
- saved metrics for both models
- saved loss curves
- saved solution comparison plot
- saved fuzzy partition visualization
- saved local loss weight visualization
- report includes explicit failure analysis
