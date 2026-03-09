---
name: tsk-pinn-experiment
description: Build, run, analyze, and iteratively improve TSK-based PINN experiments for PDE solving. Use when the task involves experiment scaffolding, model implementation, training pipelines, ablations, baselines, diagnostics, result reporting, or failure analysis for PINN, fuzzy PINN, TSK-PINN, adaptive weighting, adaptive sampling, or PDE benchmarks.
---

# TSK-PINN Experiment Orchestrator

## Purpose
This skill standardizes how Codex should design, implement, run, analyze, and refine experiments for a TSK-based PINN research project.

The skill is optimized for the following research line:
- main solver: PINN backbone
- fuzzy module: TSK-based adaptive fuzzy partition
- fuzzy output role: local reliability weighting for data loss / PDE residual loss / boundary loss
- training strategy: standard PINN optimization plus partition-aware resampling
- reporting requirement: detailed outputs, detailed interpretation, explicit failure analysis, and next-step recommendations

## When to use
Use this skill when the user asks for any of the following:
- build or modify TSK-PINN experiment code
- scaffold a PDE benchmark repo
- add baselines such as vanilla PINN or fuzzy-PINN
- run ablations on antecedent / consequent / weighting / resampling
- generate a reproducible experiment pipeline
- produce a detailed result report or diagnostics
- analyze why a training run did not meet expectations

Do not use this skill for:
- generic coding tasks unrelated to PDE experiments
- purely theoretical writing with no code or experimental workflow
- unrelated fuzzy-classification or tabular-learning projects

## Non-negotiable output rules
Every substantive experiment run must produce a report that includes all of the following:
1. exact configuration used
2. code paths or entrypoints used
3. datasets / PDEs used
4. metrics table
5. training-curve summary
6. qualitative artifact summary (plots, partition maps, local weight maps)
7. comparison against baseline(s)
8. what worked
9. what did not work
10. explicit section named `Not Meeting Expectations` whenever any metric, behavior, or visualization is unsatisfactory
11. explicit section named `Actionable Next Changes` with ranked modifications for the next iteration

Never end with only “completed” or “code written”.

## Multi-agent collaboration protocol
Default to role-based collaboration.

If true multi-agent support is available, split work across agents.
If not, simulate role-based collaboration sequentially and preserve role boundaries in the final report.

Use these roles:

### Agent A — Research Planner
Responsibilities:
- restate experiment goal precisely
- identify target PDE(s), metrics, baselines, and expected signals
- define success criteria and minimal reproducible experiment scope
- decide first-pass ablations

Deliverables:
- experiment plan
- definitions of done
- risk list

### Agent B — Model Architect
Responsibilities:
- implement or revise model code
- enforce architecture conventions for PINN backbone and TSK modules
- verify tensor shapes and computational graph correctness
- minimize unnecessary complexity in version 1

Deliverables:
- model specification
- code changes
- architecture notes

### Agent C — Training & Infra Engineer
Responsibilities:
- implement training loop, checkpointing, logging, plotting, config loading
- ensure reproducibility via random seeds
- add evaluation hooks and save artifacts
- make run scripts easy to invoke

Deliverables:
- runnable training/evaluation pipeline
- saved artifacts
- run commands

### Agent D — Evaluator & Diagnostician
Responsibilities:
- summarize final metrics
- compare against baselines and ablations
- identify failure modes
- propose next changes grounded in evidence

Deliverables:
- result report
- failure analysis
- prioritized next-step list

## Project conventions for this research line

### Task framing
Assume the paper’s first version should prioritize stability and interpretability over maximal flexibility.

### Recommended v1 model design
Use the following as the default v1 design unless the user overrides it.

#### Backbone PINN
- use a modified fully-connected network (mFCN) style backbone if practical
- default hidden width: 128
- default hidden depth: 5
- default activation: tanh
- optional Fourier features for oscillatory or multi-scale PDEs
- default optimization schedule: Adam pretraining then L-BFGS refinement

#### TSK antecedent
- use Gaussian antecedents
- antecedent input should emphasize geometry/state variables rather than high-dimensional hidden states
- include normalized coordinates and distance-to-boundary / distance-to-initial-surface when available
- use a small linear projection before the membership computation
- stabilize antecedent outputs with normalization if needed
- avoid numerically fragile product-style implementations if they cause underflow
- initialize centers with K-means or FCM if available
- use a regularizer that discourages winner-take-all rule collapse

#### TSK consequent
- version 1 uses first-order consequent functions
- consequent input should be a compact bottleneck feature derived from the backbone hidden representation
- consequent outputs should parameterize three local loss logits:
  - data
  - PDE
  - boundary / initial
- final local weights should be normalized pointwise via softmax

#### Weighted losses
Default total loss structure:
- weighted data loss
- weighted PDE residual loss
- weighted boundary/initial loss
- regularization for partition balance and entropy if implemented

### Partition-aware resampling
For adaptive sampling, use partition-level difficulty statistics rather than only isolated point residuals when possible.

### Baselines to include when feasible
At minimum:
- vanilla PINN
- proposed TSK-PINN

Then add as feasible:
- fuzzy-PINN / DFPINN-style representation baseline
- adaptive weighting only baseline
- adaptive sampling only baseline

## Required repository outputs
A completed experiment pass should create or update the following when practical:
- `configs/` YAML or TOML experiment configs
- `src/` implementation modules
- `scripts/` run scripts
- `outputs/` or `runs/` with:
  - metrics
  - logs
  - plots
  - checkpoints
  - report markdown

## Required report structure
Use the report template in `templates/RESULT_REPORT_TEMPLATE.md`.

At minimum include:
- Goal
- Setup
- Code changes
- Exact commands run
- Metrics
- Visual findings
- Baseline comparison
- Not Meeting Expectations
- Probable Causes
- Actionable Next Changes

## Definition of done for v1
Version 1 is done only if all of the following are true:
1. a minimal PDE benchmark runs end-to-end
2. vanilla PINN baseline is implemented
3. TSK-PINN variant is implemented
4. all runs save reproducible configs and metrics
5. at least one comparison report exists
6. failure modes are explicitly documented if performance is not improved

## Safety / honesty rules
- Never claim improvement without actual measured comparison.
- If the proposed model underperforms, say so plainly.
- If a plot or metric is missing, say exactly what is missing and why.
- If the experiment did not converge, document the evidence and suspected root causes.
- Distinguish clearly between verified results and hypotheses.

## Execution order
When asked to perform an experiment task, follow this order:
1. restate the target experiment
2. inspect repository structure
3. write or update configs before writing large code blocks
4. implement the smallest runnable version first
5. run or prepare baseline first
6. run or prepare proposed method second
7. compare outputs
8. write the report
9. list what to try next

## First-pass benchmark recommendation
Unless the user explicitly overrides, start with:
- PDE: 1D Burgers equation
- models: vanilla PINN vs TSK local-reliability-weighted PINN
- outputs: training curves, final error table, fuzzy partition visualization, local weight visualization, failure analysis

## Evaluation guidance
Use the prompts in `evals/prompts.csv` to test whether this skill triggers correctly and whether the output quality is acceptable.
