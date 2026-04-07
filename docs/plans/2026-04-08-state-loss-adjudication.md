# State-Loss Warmstart Adjudication

## Scope

This note records the final adjudication for the `state-loss-warmstart-2000-20260407` review.

Authoritative runtime:

- host: `dell@192.168.1.104`
- repo: `/home/dell/work/le-wm-state-loss-clean-20260407`
- venv: `~/venvs/lewm`
- env: `STABLEWM_HOME=~/stablewm`

Authoritative evaluator anchor:

- package file: `/home/dell/venvs/lewm/lib/python3.10/site-packages/stable_worldmodel/world.py`
- function: `stable_worldmodel.World.evaluate_from_dataset(...)`
- function line observed during adjudication: `755`

The conclusion below is anchored to the original installed evaluator path above, not to helper paths such as `evaluate_from_dataset_local`, `evaluate_from_dataset_with_history`, `independent_solver`, `deterministic`, or `chunk_size` experiments.

## Final Verdict

| Case bucket | Episode | Start step | `exp_0002` | `exp_0006` | Final interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| residual | `16015` | `94` | `0/5` | `5/5` | reject `exp_0002` success claim |
| residual | `16637` | `6` | `0/5` | `5/5` | stable `exp_0006` win |
| swing | `5865` | `66` | `5/5` | `5/5` | not a stable regression |
| swing | `9864` | `24` | `5/5` | `5/5` | not a stable regression |
| swing | `2860` | `57` | `5/5` | `5/5` | not a stable regression |
| swing | `6651` | `31` | `5/5` | `5/5` | not a stable regression |
| swing | `12473` | `7` | `5/5` | `5/5` | not a stable regression |

## Key Decision

`16015:94 / exp_0002` must be counted as `0/5`, not `1/5`.

Reason:

- an older accepted artifact showed only `seed=4` as positive
- a later fresh rerun showed only `seed=3` as positive
- isolated fresh-process reruns for `seed=4` were false twice
- isolated fresh-process reruns for `seed=3` were false three times

This pattern is solver or runtime instability, not a reproducible success.

## Evidence

Primary artifacts:

- accepted multiseed snapshot: `/home/dell/stablewm/autoresearch/pusht/state-loss-warmstart-2000-20260407/remaining_case_multiseed_clean_20260408_callables.json`
- fresh rerun snapshot: `/home/dell/stablewm/autoresearch/pusht/state-loss-warmstart-2000-20260407/remaining_case_multiseed_rerun_16015_exp0002_fresh_20260408.json`

Supporting facts already established during adjudication:

- isolated `16015:94 / exp_0002 / seed=4` reruns: `false`, `false`
- isolated `16015:94 / exp_0002 / seed=3` reruns: `false`, `false`, `false`

Interpretation rule used for the final table:

- reproducible multiseed wins count
- non-reproducible one-off positives do not count

## Bottom Line

For the residual cases, the defensible final reading is:

- `exp_0002`: only `16637:6` remains a loss; `16015:94` is also a loss after repro filtering
- `exp_0006`: both residual cases are stable wins

So the final residual adjudication is `exp_0002 = 0/5` and `exp_0006 = 5/5`.
