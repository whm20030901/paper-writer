# Experiment Result Report

## 1. Goal
- Research objective:
- PDE / dataset:
- Hypothesis being tested:
- Success criteria:

## 2. Environment and Configuration
- Date/time:
- Commit hash or working tree state:
- Device:
- Seed(s):
- Main config file:
- Model variants:
- Training schedule:

## 3. Exact Commands
```bash
# paste commands here
```

## 4. Code Changes
- Files added:
- Files modified:
- Why each change was needed:

## 5. Models Evaluated
### 5.1 Baseline(s)
- Vanilla PINN:
- Other baseline(s):

### 5.2 Proposed Model
- Backbone:
- TSK antecedent:
- TSK consequent:
- Local weighting method:
- Resampling method:

## 6. Metrics Summary
| Model | PDE | Seed | Rel L2 Error | PDE Residual | Boundary Error | Train Time | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| | | | | | | | |

## 7. Training Dynamics
- Did training converge?
- Did the baseline converge?
- Did the proposed method converge?
- Any instability, oscillation, saturation, or collapse?

## 8. Visual Findings
- Available plots:
- What the fuzzy partition visualization shows:
- What the local loss weight visualization shows:
- Whether partitions are balanced or collapsed:

## 9. Baseline Comparison
- Where the proposed model is better:
- Where it is worse:
- Whether gains are statistically meaningful or only anecdotal:

## 10. Not Meeting Expectations
List every unsatisfactory item separately.

### Item A
- Expected:
- Observed:
- Evidence:
- Severity:

### Item B
- Expected:
- Observed:
- Evidence:
- Severity:

## 11. Probable Causes
For each failed or weak outcome, give evidence-backed hypotheses.

### Cause 1
- Linked issue:
- Why this is plausible:
- What evidence supports it:
- What evidence is still missing:

## 12. Actionable Next Changes
Rank from highest value to lowest value.

1. 
   - Why:
   - Cost:
   - Expected effect:
2. 
   - Why:
   - Cost:
   - Expected effect:
3. 
   - Why:
   - Cost:
   - Expected effect:

## 13. Final Assessment
- Is this iteration publishable-quality? Why or why not?
- Should the next iteration focus on architecture, training, sampling, or diagnostics?
