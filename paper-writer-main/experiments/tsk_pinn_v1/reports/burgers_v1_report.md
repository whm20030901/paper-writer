# Experiment Result Report

## 1. Goal
- Research objective: 搭建并跑通 1D Burgers 方程的最小可复现实验（Vanilla PINN vs TSK-PINN）。
- PDE / dataset: 无监督 PINN 训练点（内部点 + 边界点 + 初始点），Burgers 方程。
- Hypothesis being tested: TSK 的局部可靠性加权可改善训练行为或最终误差。
- Success criteria: 两个模型端到端可运行，产出指标与可视化。

## 2. Environment and Configuration
- Date/time: 由执行时填充。
- Commit hash or working tree state: 由执行时填充。
- Device: CPU（默认）。
- Seed(s): 42。
- Main config file:
  - `experiments/tsk_pinn_v1/configs/burgers_baseline.yaml`
  - `experiments/tsk_pinn_v1/configs/burgers_tsk.yaml`
- Model variants:
  - Vanilla PINN
  - TSK local reliability weighted PINN
- Training schedule: Adam，300 epochs（可在 YAML 调整）。

## 3. Exact Commands
```bash
python experiments/tsk_pinn_v1/scripts/train_baseline.py --config experiments/tsk_pinn_v1/configs/burgers_baseline.yaml
python experiments/tsk_pinn_v1/scripts/train_tsk.py --config experiments/tsk_pinn_v1/configs/burgers_tsk.yaml
python experiments/tsk_pinn_v1/scripts/eval_compare.py \
  --baseline experiments/tsk_pinn_v1/outputs/burgers_baseline \
  --tsk experiments/tsk_pinn_v1/outputs/burgers_tsk \
  --out experiments/tsk_pinn_v1/outputs/compare/metrics_compare.csv
```

## 4. Code Changes
- Files added: configs/scripts/src/reports 全套最小实验骨架。
- Why each change was needed:
  - 保证 baseline 与 proposed 有独立入口。
  - 保证指标、曲线、可视化、checkpoint 可落盘。
  - 保证可复现（seed + config 驱动）。

## 5. Models Evaluated
### 5.1 Baseline(s)
- Vanilla PINN: MLPBackbone + 标准 `L_pde + L_bc + L_ic`。

### 5.2 Proposed Model
- Backbone: 与 baseline 同构 MLP。
- TSK antecedent: Gaussian antecedent。
- TSK consequent: 一阶 consequent，输出 3 个损失 logits。
- Local weighting method: 对 `data/pde/bc` 点级 softmax 权重。
- Resampling method: v1 暂未加入分区重采样（后续迭代项）。

## 6. Metrics Summary
运行后由 `metrics.json` 与 `metrics_compare.csv` 自动填充。

## 7. Training Dynamics
运行后补充是否收敛、是否震荡。

## 8. Visual Findings
目标产物：
- loss curves
- solution compare
- partition map
- local weights (λ_d, λ_p, λ_b)

## 9. Baseline Comparison
依据 `outputs/compare/metrics_compare.csv` 填写。

## 10. Not Meeting Expectations
### Item A
- Expected: TSK 至少不劣于 baseline。
- Observed: 待运行验证。
- Evidence: 待补。
- Severity: 待评估。

## 11. Probable Causes
- 可能原因：规则数不足/过多、熵正则不合适、训练轮次不足、参考解离散误差。

## 12. Actionable Next Changes
1. 加入 rule 数量 ablation（4/6/8）。
2. 加入 bottleneck 维度 ablation（4/8/12）。
3. 加入 DropRule 与 uniform-regularization 对照。

## 13. Final Assessment
- 当前为可运行 v1 骨架，需执行完整实验并写入实测结果后，才可评估是否达到可发布质量。
