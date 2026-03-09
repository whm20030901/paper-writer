# Experiment Spec: TSK-PINN V1

## Research question
Can a TSK-based local reliability weighting mechanism improve training behavior or final accuracy relative to a vanilla PINN on a simple PDE benchmark?

## Model summary
### Baseline
- modified fully-connected PINN backbone
- standard weighted sum of data / PDE / boundary losses

### Proposed model
- same backbone as baseline
- Gaussian TSK antecedent over geometry/state inputs
- first-order TSK consequent over compact hidden features
- pointwise local weights for data / PDE / boundary losses
- optional partition-aware resampling after baseline pipeline is stable

## Metrics
- relative L2 error
- PDE residual MSE
- boundary/initial MSE
- wall-clock training time
- convergence observations

## Diagnostics to save
- total loss curve
- per-loss curve
- predicted vs reference solution plot
- fuzzy partition heatmap
- local weight map for λ_d, λ_p, λ_b
- rule utilization histogram

## Initial ablations
- number of rules R ∈ {4, 6, 8}
- consequent bottleneck dimension dc ∈ {4, 8, 12}
- DropRule on/off
- uniform-regularization on/off
