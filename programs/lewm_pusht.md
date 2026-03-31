# LeWM PushT Program

## Objective

Improve official `PushT` `success_rate` under the repository's existing benchmark surface.

## Benchmark Contract

- Training entrypoint is `train.py`.
- Official evaluation entrypoint is `eval.py --config-name=pusht.yaml`.
- Evaluation config lives in `config/eval/pusht.yaml`.
- Bounded candidate runs must force `data=pusht`.
- Bounded candidate runs must force `wandb.enabled=False`.
- Bounded candidate runs must force `+trainer.max_steps=2000` unless a smaller smoke budget is explicitly requested.
- The only score that matters is official planner `success_rate`.

## Mutable Files

Only these model-side files may be changed when forming a candidate:

- `train.py`
- `jepa.py`
- `module.py`

## Run Status

- `keep`: candidate completed and strictly improved official `success_rate`.
- `discard`: candidate completed but did not improve official `success_rate`.
- `crash`: candidate did not produce usable metrics.

## Stop Conditions

- stop after `20` total experiments
- stop after `3` consecutive crashes
- stop after `8` consecutive non-improving runs
- stop once `success_rate >= 90.0`

## Current Baseline

- official pretrained `success_rate = 98.0`
- official eval time is about `488s`
- bounded training throughput is about `4.1 steps/s`

## Operator Loop

1. Edit one small hypothesis in `train.py`, `jepa.py`, or `module.py`.
2. Run `python3 tools/autoresearch_loop.py --tag <tag> --description "..."` once the outer loop exists.
3. Read `keep`, `discard`, or `crash`.
4. Continue from the updated best state in the active AI session.
