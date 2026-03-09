# TSK-PINN V1 Experiment Skeleton

## Objective
Build the first minimal, reproducible benchmark for a TSK-based local reliability weighted PINN.

## Layout
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
    data/burgers.py
    models/{backbone.py,tsk.py,losses.py}
    trainers/train.py
    utils/io.py
  outputs/
  reports/
    burgers_v1_report.md
```

## Quick start
```bash
python experiments/tsk_pinn_v1/scripts/train_baseline.py --config experiments/tsk_pinn_v1/configs/burgers_baseline.yaml
python experiments/tsk_pinn_v1/scripts/train_tsk.py --config experiments/tsk_pinn_v1/configs/burgers_tsk.yaml
python experiments/tsk_pinn_v1/scripts/eval_compare.py \
  --baseline experiments/tsk_pinn_v1/outputs/burgers_baseline \
  --tsk experiments/tsk_pinn_v1/outputs/burgers_tsk \
  --out experiments/tsk_pinn_v1/outputs/compare/metrics_compare.csv
```

## Notes
- 该版本是最小可运行实验骨架（baseline + TSK + 指标/可视化保存）。
- 不应在未实测比较前宣称性能提升。
- 若 TSK 表现不佳，请在报告中明确写入 `Not Meeting Expectations`。
