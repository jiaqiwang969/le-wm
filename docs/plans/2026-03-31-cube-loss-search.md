# Cube Loss Search Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a bounded `mse_cosine` prediction loss option and evaluate whether it improves official `cube` success rate over the current `500-step + adaln + lr=1e-4` baseline.

**Architecture:** Keep model structure, inference, planner behavior, and checkpoint format unchanged. Limit the change to the training loss path in `module.py`, wire the new knobs through `train.py` and `config/train/lewm.yaml`, prove correctness with focused unit tests, then run the same remote train/eval pipeline already established for `cube`.

**Tech Stack:** Python, PyTorch, Hydra, pytest/unittest, remote SSH execution on `dell@192.168.1.104`

---

### Task 1: Add failing tests for the new loss mode

**Files:**
- Modify: `tests/test_searchable_model_knobs.py`

**Step 1: Write the failing tests**

Add tests that pin down the expected `mse_cosine` behavior.

```python
def test_prediction_loss_supports_mse_cosine(self):
    pred = torch.tensor([[1.0, 0.0]])
    target = torch.tensor([[0.0, 1.0]])

    loss = prediction_loss(
        pred,
        target,
        loss_type="mse_cosine",
        mse_weight=1.0,
        cosine_weight=0.1,
    )

    expected = torch.tensor(1.0 + 0.1)
    self.assertTrue(torch.isclose(loss, expected))

def test_prediction_loss_mse_cosine_can_detach_target(self):
    pred = torch.tensor([[1.0, 0.0]], requires_grad=True)
    target = torch.tensor([[0.0, 1.0]], requires_grad=True)

    loss = prediction_loss(
        pred,
        target,
        loss_type="mse_cosine",
        target_detach=True,
        mse_weight=1.0,
        cosine_weight=0.1,
    )
    loss.backward()

    self.assertIsNone(target.grad)
```

**Step 2: Run test to verify it fails**

Run:

```bash
cd /Users/jqwang/09-世界模型-操控/le-wm
pytest tests/test_searchable_model_knobs.py -q
```

Expected:

- FAIL because `prediction_loss()` does not yet accept `mse_weight` / `cosine_weight`
- or FAIL because `loss_type="mse_cosine"` is unsupported

**Step 3: Commit**

Do not commit yet. This task should leave the repo with failing tests only.

### Task 2: Implement `mse_cosine` in `prediction_loss`

**Files:**
- Modify: `module.py`

**Step 1: Write the minimal implementation**

Extend `prediction_loss()` with two new keyword args:

- `mse_weight=1.0`
- `cosine_weight=0.1`

Implement:

```python
if loss_type == "mse_cosine":
    mse = F.mse_loss(pred, target)
    cosine = 1 - F.cosine_similarity(pred, target, dim=-1).mean()
    return mse_weight * mse + cosine_weight * cosine
```

Keep existing behavior unchanged for:

- `mse`
- `smooth_l1`
- unknown `loss_type` error path

Preserve `target_detach` before all loss branches.

**Step 2: Run the focused test file**

Run:

```bash
cd /Users/jqwang/09-世界模型-操控/le-wm
pytest tests/test_searchable_model_knobs.py -q
```

Expected:

- PASS for old tests
- PASS for new `mse_cosine` tests

**Step 3: Commit**

```bash
git add module.py tests/test_searchable_model_knobs.py
git commit -m "feat: add mse cosine prediction loss"
```

### Task 3: Wire the new loss knobs through training config

**Files:**
- Modify: `train.py`
- Modify: `config/train/lewm.yaml`

**Step 1: Add config defaults**

Under `loss.pred` in `config/train/lewm.yaml`, add:

```yaml
    mse_weight: 1.0
    cosine_weight: 0.1
```

Do not change the default `type: mse`.

**Step 2: Pass the new config fields into `prediction_loss()`**

Update the call in `train.py`:

```python
output["pred_loss"] = prediction_loss(
    pred_emb,
    tgt_emb,
    loss_type=cfg.loss.pred.type,
    target_detach=cfg.loss.pred.target_detach,
    smooth_l1_beta=cfg.loss.pred.smooth_l1_beta,
    mse_weight=cfg.loss.pred.mse_weight,
    cosine_weight=cfg.loss.pred.cosine_weight,
)
```

**Step 3: Run tests again**

Run:

```bash
cd /Users/jqwang/09-世界模型-操控/le-wm
pytest tests/test_searchable_model_knobs.py -q
```

Expected:

- PASS

**Step 4: Commit**

```bash
git add train.py config/train/lewm.yaml
git commit -m "feat: expose configurable mse cosine loss"
```

### Task 4: Run local verification for the new knobs

**Files:**
- No new files required

**Step 1: Verify unit tests for searchable knobs**

Run:

```bash
cd /Users/jqwang/09-世界模型-操控/le-wm
pytest tests/test_searchable_model_knobs.py -q
```

Expected:

- PASS

**Step 2: Verify full autoresearch-related tests still pass**

Run:

```bash
cd /Users/jqwang/09-世界模型-操控/le-wm
pytest tests/test_autoresearch_tools.py tests/test_autoresearch_loop.py -q
```

Expected:

- PASS

**Step 3: Dry-run the train command locally to confirm config wiring**

Run:

```bash
cd /Users/jqwang/09-世界模型-操控/le-wm
python3 tools/run_candidate.py \
  --benchmark cube \
  --tag plancheck \
  --exp-id dryrun_mse_cosine \
  --max-steps 500 \
  --dry-run \
  --override optimizer.lr=1e-4 \
  --override loss.pred.type=mse_cosine \
  --override loss.pred.mse_weight=1.0 \
  --override loss.pred.cosine_weight=0.1
```

Expected:

- JSON output contains the new overrides
- no argument or config-key errors

**Step 4: Commit**

```bash
git add module.py train.py config/train/lewm.yaml tests/test_searchable_model_knobs.py
git commit -m "test: verify mse cosine loss wiring"
```

### Task 5: Run remote `cube` fast-screen experiments

**Files:**
- Remote outputs only under `/home/dell/stablewm/autoresearch/cube/`

**Step 1: Run candidate A on the remote server**

Run:

```bash
ssh dell@192.168.1.104 '
  source ~/venvs/lewm/bin/activate &&
  cd ~/work/le-wm &&
  export STABLEWM_HOME=/home/dell/stablewm &&
  python3 tools/run_candidate.py \
    --benchmark cube \
    --tag losssearch \
    --exp-id exp_0001 \
    --max-steps 500 \
    --repo-root /home/dell/work/le-wm \
    --stablewm-home /home/dell/stablewm \
    --override optimizer.lr=1e-4 \
    --override loss.pred.type=mse_cosine \
    --override loss.pred.mse_weight=1.0 \
    --override loss.pred.cosine_weight=0.1 &&
  python3 tools/eval_candidate.py \
    --benchmark cube \
    --tag losssearch \
    --exp-id exp_0001 \
    --repo-root /home/dell/work/le-wm \
    --stablewm-home /home/dell/stablewm \
    --override eval.num_eval=10 \
    --override eval.goal_offset_steps=10 \
    --override eval.eval_budget=25 \
    --override seed=42
'
```

**Step 2: Run candidate B on the remote server**

Run the same command with:

```bash
--exp-id exp_0002
--override loss.pred.cosine_weight=0.25
```

**Step 3: Update the results table**

Run:

```bash
ssh dell@192.168.1.104 '
  source ~/venvs/lewm/bin/activate &&
  cd ~/work/le-wm &&
  export STABLEWM_HOME=/home/dell/stablewm &&
  python3 tools/update_results.py \
    --benchmark cube \
    --tag losssearch \
    --exp-id exp_0001 \
    --repo-root /home/dell/work/le-wm \
    --stablewm-home /home/dell/stablewm \
    --description "500-step adaln lr1e-4 mse_cosine cw0.1 eval10 goal10 budget25 seed42" &&
  python3 tools/update_results.py \
    --benchmark cube \
    --tag losssearch \
    --exp-id exp_0002 \
    --repo-root /home/dell/work/le-wm \
    --stablewm-home /home/dell/stablewm \
    --description "500-step adaln lr1e-4 mse_cosine cw0.25 eval10 goal10 budget25 seed42"
'
```

**Step 4: Apply the fast-screen gate**

Keep only candidates with:

- fast eval `>= 80%`

Drop anything below that threshold.

**Step 5: Commit**

No local code commit here. This is an experiment run checkpoint only.

### Task 6: Run official evaluation only for surviving candidates

**Files:**
- Remote results under each experiment directory

**Step 1: Run official eval for each survivor**

Example command:

```bash
ssh dell@192.168.1.104 '
  source ~/venvs/lewm/bin/activate &&
  cd ~/work/le-wm &&
  export STABLEWM_HOME=/home/dell/stablewm &&
  python3 eval.py \
    --config-name=cube.yaml \
    policy=autoresearch/cube/losssearch/exp_0001/lewm_epoch_1 \
    output.filename=ogb_cube_results_official.txt
'
```

**Step 2: Compare against the current official baseline**

Current baseline to beat:

- `500-step + adaln + lr=1e-4`
- official mean `57%` across `seed42` and `seed123`

**Step 3: If a candidate improves on one official seed, run a second seed**

Run:

```bash
ssh dell@192.168.1.104 '
  source ~/venvs/lewm/bin/activate &&
  cd ~/work/le-wm &&
  export STABLEWM_HOME=/home/dell/stablewm &&
  python3 eval.py \
    --config-name=cube.yaml \
    policy=autoresearch/cube/losssearch/exp_0001/lewm_epoch_1 \
    output.filename=ogb_cube_results_official_seed123.txt \
    seed=123
'
```

**Step 4: Decide keep / stop**

Keep the loss family only if:

- official mean `> 57%`, or
- official mean `== 57%` with a clear efficiency advantage

Otherwise stop this loss family and move to the next search axis.
