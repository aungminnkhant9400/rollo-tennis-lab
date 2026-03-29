# Research Policy

## Objective

The objective of this project is to improve tennis match winner prediction through small, controlled, and well-evaluated research iterations.

## Task Definition

The task is tennis match winner prediction. All research and code changes must serve that objective directly.

## Core Rules

- Evaluation discipline is critical.
- Future data leakage is forbidden.
- Evaluation code and split logic must not be modified by the agent unless the human explicitly allows it.
- Only small, justified changes should be proposed.
- Simplicity is preferred over complexity.
- Every claim must be supported by measured results.

## Experiment Standard

Each experiment must include:

- hypothesis
- files changed
- expected effect
- result

## Agent Behavior

- Do not weaken evaluation to make results look better.
- Do not use information from the future when building features, labels, or splits.
- Do not mix exploratory edits with broad refactors.
- If a change affects validity, stop and require explicit human approval.
- Prefer clear, reviewable steps over ambitious changes.
