# Cube Loss Search Design

**Date:** 2026-03-31

**Goal:** Improve official `cube` benchmark success rate with a minimal, attributable training change.

## Context

Recent `cube` experiments converged on a stable baseline:

- `500-step`
- `predictor.conditioning_type=adaln`
- `optimizer.lr=1e-4`

Official evaluation showed that:

- `lr=1e-4` is the main source of improvement over the default `5e-5`
- `conditioning_type=add` is less stable than `adaln`
- `1000-step` training is not currently worth the extra time

This makes the training loss the cleanest next search axis. It changes the learned latent geometry without changing inference, planner behavior, or checkpoint interfaces.

## Non-Goals

- Do not change `eval.py`, planner settings, or dataset wiring
- Do not change checkpoint format or policy loading behavior
- Do not mix model-structure changes with loss changes in the same round
- Do not broaden the search space yet beyond one bounded loss family

## Proposed Change

Extend `prediction_loss` with a new option: `mse_cosine`.

Formula:

`loss = mse_weight * MSE(pred, target) + cosine_weight * mean(1 - cosine_similarity(pred, target))`

This preserves the current scale-sensitive MSE term while adding an angular alignment term for latent targets. The goal is to encourage directional consistency in embedding space without replacing the current objective entirely.

## Files In Scope

- `module.py`
- `train.py`
- `config/train/lewm.yaml`
- `tests/test_searchable_model_knobs.py`

## Files Out Of Scope

- `jepa.py`
- `eval.py`
- planner / solver configs
- autoresearch orchestration tools

## Configuration Design

Keep existing defaults unchanged.

Add new prediction-loss config fields under `loss.pred`:

- `mse_weight: 1.0`
- `cosine_weight: 0.1`

Supported `loss.pred.type` values:

- `mse`
- `smooth_l1`
- `mse_cosine`

For `mse` and `smooth_l1`, the new weights are ignored.

## Experiment Protocol

Baseline remains:

- `500-step`
- `adaln`
- `lr=1e-4`

Candidate loss settings:

1. `loss.pred.type=mse_cosine`, `loss.pred.mse_weight=1.0`, `loss.pred.cosine_weight=0.1`
2. `loss.pred.type=mse_cosine`, `loss.pred.mse_weight=1.0`, `loss.pred.cosine_weight=0.25`

Evaluation sequence:

1. Run fast screen with `eval.num_eval=10`, `goal_offset_steps=10`, `eval.eval_budget=25`, `seed=42`
2. Drop any candidate below `80%` fast-eval success
3. For surviving candidates, run official `cube.yaml` evaluation
4. If official single-seed improves, run a second official seed before keeping it

## Success Criteria

- Fast screen is not worse than the current `80%` stable tier
- Official evaluation is at least competitive with the current `56-58%` range
- Keep only candidates whose multi-seed mean exceeds the current `57%` baseline, or ties it with a meaningful efficiency advantage

## Risk Assessment

Primary risks:

- cosine term may overweight direction and weaken magnitude information
- fast-screen improvements may not transfer to official evaluation
- too-large cosine weights may destabilize optimization

Risk controls:

- start with small cosine weights only
- keep architecture fixed
- require official confirmation before accepting a change

## Rollback Rule

If `mse_cosine` shows no clear official improvement after the two initial weights, stop this loss family and move on to the next search axis instead of continuing to tune it incrementally.
