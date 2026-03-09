# Start Here: TSK-PINN Codex Kit

This kit gives you a Codex-ready starting point for a TSK-PINN research workflow.

## What is included
- `.codex/skills/tsk-pinn-experiment/SKILL.md` — the specialized Codex skill
- report and checklist templates
- skill-eval prompts
- a starter experiment folder for `experiments/tsk_pinn_v1`
- ready-to-paste Codex prompts

## Recommended first task
Start with a minimal reproducible experiment on the 1D Burgers equation:
- baseline: vanilla PINN
- proposed: TSK-based local reliability weighted PINN
- outputs: metrics, plots, partition map, local weight map, failure analysis

## Recommended repo placement
Copy the `.codex` folder into the root of your working repository.
Keep the `experiments/tsk_pinn_v1` folder in the same repository root.

## Result requirement
Do not accept a run unless it produces a markdown report with:
- what improved
- what did not improve
- evidence
- next changes
