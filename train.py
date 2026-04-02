import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from freeze_modules import apply_freeze_configuration
from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg, prediction_loss
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack
from warm_start import apply_warm_start, resolve_warm_start_checkpoint_path


def prediction_loss_kwargs(cfg):
    pred_cfg = cfg.loss.pred
    return {
        "loss_type": pred_cfg.type,
        "target_detach": pred_cfg.target_detach,
        "smooth_l1_beta": pred_cfg.smooth_l1_beta,
        "mse_weight": pred_cfg.get("mse_weight", 1.0),
        "cosine_weight": pred_cfg.get("cosine_weight", 0.1),
    }


def teacher_forced_targets(emb, act_emb, *, history_size, shift):
    if shift < 1:
        raise ValueError(f"teacher forcing shift must be >= 1, got {shift}")
    required_steps = history_size + shift
    if emb.size(1) < required_steps or act_emb.size(1) < history_size:
        raise ValueError(
            f"sequence too short for teacher forcing: need emb >= {required_steps} and act >= {history_size}, "
            f"got emb={emb.size(1)} act={act_emb.size(1)}"
        )

    ctx_emb = emb[:, :history_size]
    ctx_act = act_emb[:, :history_size]
    tgt_emb = emb[:, shift : shift + history_size]
    return ctx_emb, ctx_act, tgt_emb


def autoregressive_rollout_predictions(model, emb, act_emb, *, history_size, rollout_steps):
    if rollout_steps < 1:
        raise ValueError(f"rollout_steps must be >= 1, got {rollout_steps}")
    if emb.size(1) < history_size:
        raise ValueError(
            f"sequence too short for rollout history: need emb >= {history_size}, got emb={emb.size(1)}"
        )
    required_actions = history_size + rollout_steps - 1
    if act_emb.size(1) < required_actions:
        raise ValueError(
            f"sequence too short for rollout actions: need act >= {required_actions}, got act={act_emb.size(1)}"
        )

    hist_emb = emb[:, :history_size]
    hist_act = act_emb[:, :history_size]
    preds = []

    for step in range(rollout_steps):
        next_pred = model.predict(
            hist_emb[:, -history_size:],
            hist_act[:, -history_size:],
        )[:, -1:]
        preds.append(next_pred)
        hist_emb = torch.cat([hist_emb, next_pred], dim=1)

        if step + 1 < rollout_steps:
            next_action = act_emb[:, history_size + step : history_size + step + 1]
            hist_act = torch.cat([hist_act, next_action], dim=1)

    return torch.cat(preds, dim=1)


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    rollout_steps = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight
    pred_cfg = cfg.loss.pred
    teacher_force_shift = pred_cfg.get("teacher_force_shift", rollout_steps)
    rollout_weight = pred_cfg.get("rollout_weight", 0.0)

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb, ctx_act, tgt_emb = teacher_forced_targets(
        emb,
        act_emb,
        history_size=ctx_len,
        shift=teacher_force_shift,
    )
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    # LeWM loss
    output["pred_loss"] = prediction_loss(
        pred_emb,
        tgt_emb,
        **prediction_loss_kwargs(cfg),
    )
    output["rollout_loss"] = emb.new_zeros(())
    if rollout_weight > 0:
        rollout_pred = autoregressive_rollout_predictions(
            self.model,
            emb,
            act_emb,
            history_size=ctx_len,
            rollout_steps=rollout_steps,
        )
        rollout_tgt = emb[:, ctx_len : ctx_len + rollout_steps]
        output["rollout_loss"] = prediction_loss(
            rollout_pred,
            rollout_tgt,
            **prediction_loss_kwargs(cfg),
        )

    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = (
        output["pred_loss"]
        + rollout_weight * output["rollout_loss"]
        + lambd * output["sigreg_loss"]
    )

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


def maybe_apply_warm_start(model, cfg, stablewm_home: Path):
    warm_start_cfg = cfg.get("warm_start")
    if not warm_start_cfg or not warm_start_cfg.get("enabled", False):
        return None

    checkpoint_path = resolve_warm_start_checkpoint_path(
        stablewm_home=stablewm_home,
        checkpoint=warm_start_cfg.get("checkpoint"),
    )
    print(f"Loading warm-start checkpoint from {checkpoint_path}")
    result = apply_warm_start(
        model,
        checkpoint_path,
        strict=warm_start_cfg.get("strict", True),
    )
    print("Warm-start checkpoint loaded")
    return result

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(
        input_dim=effective_act_dim,
        emb_dim=embed_dim,
        **cfg.action_encoder,
    )
    
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=cfg.projector.hidden_dim,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=cfg.predictor_proj.hidden_dim,
        norm_fn=torch.nn.BatchNorm1d,
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )

    stablewm_home = Path(swm.data.utils.get_cache_dir())
    maybe_apply_warm_start(world_model, cfg, stablewm_home=stablewm_home)
    frozen_modules = apply_freeze_configuration(world_model, cfg.get("freeze"))
    if frozen_modules:
        print(f"Frozen modules: {', '.join(frozen_modules)}")

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()
    if cfg.get("post_fit_validate", False):
        print("Running post-fit validation on final weights")
        trainer.validate(model=world_model, dataloaders=val, verbose=True)
    return


if __name__ == "__main__":
    run()
