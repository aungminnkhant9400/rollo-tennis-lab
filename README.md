# rollo-tennis-lab

Clean research workspace for tennis match winner prediction.

## Project Goal

Build a disciplined prediction pipeline for tennis match winner forecasting, with a long-term path toward an autoresearch loop that can propose, run, and review bounded experiments.

## Scope of v1

Version 1 is focused on a simple and professional foundation:

- define the research problem clearly
- keep evaluation rules strict
- support small, reviewable experiments
- separate editing from remote experiment execution

## Planned File Structure

```text
rollo-tennis-lab/
|- README.md
|- program.md
|- data/
|- src/
|- experiments/
`- outputs/
```

## Basic Workflow

- Edit code, notes, and experiment setup on a Windows laptop.
- Sync the project to an A100 Linux server.
- Run training and evaluation on the Linux server.
- Review results, keep useful changes, and discard weak ideas.

## Not In Scope Yet

- automated experiment orchestration
- large-scale hyperparameter search
- production deployment
- dashboards and reporting layers
- complex infrastructure
