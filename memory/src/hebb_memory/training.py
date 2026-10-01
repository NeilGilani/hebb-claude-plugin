"""Train the memory for a base model, once. About an hour on a consumer GPU for a 0.5B model.

What is trained is small: a key head that turns a sentence into an address, read heads that
let the frozen model attend to pages, and the page initialisation and write step. The base
model's own weights are never touched. The result goes into the registry, where `attach`
finds it by model id.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .registry import PRODUCT, RECIPE, Registry, product_config

#: Language-pretraining steps for the toy model. Training and attaching must rebuild the same
#: toy, so both read this.
TOY_PRETRAIN = 300

KNOBS = ("MODEL", "DEVICE", "KEY_STEPS", "KEY_SLOTS", "FREEZE_KEYS", "CUE_KIND", "META_EPISODES",
         "META_INNER", "META_CKPT")


def train(model_id: str, *, registry: Optional[Registry] = None, device: str = "cpu",
          pretrain_steps: Optional[int] = None, ckpt_dir: Optional[Path | str] = None, note: str = "",
          **recipe_overrides) -> Path:
    """Train with the paper's recipe and register the result.

    `model_id="toy"` trains a tiny model on the CPU in seconds (tests, trying the API). Anything
    else is a Hugging Face model id. `ckpt_dir` makes training resumable if it is interrupted.
    `pretrain_steps` only applies to the toy model, which has to learn English first.
    """
    from ._train import meta as R
    pretrain_steps = TOY_PRETRAIN if pretrain_steps is None else pretrain_steps
    recipe = {**RECIPE, **recipe_overrides}
    saved = {k: getattr(R, k) for k in KNOBS}
    try:
        R.MODEL = None if model_id == "toy" else model_id
        R.DEVICE = device
        R.KEY_STEPS, R.KEY_SLOTS, R.FREEZE_KEYS = recipe["key_steps"], recipe["key_slots"], recipe["freeze_keys"]
        R.CUE_KIND, R.META_EPISODES = PRODUCT["cue_kind"], recipe["meta_episodes"]
        R.META_INNER = tuple(recipe["meta_inner"])
        R.META_CKPT = Path(ckpt_dir) if ckpt_dir else None
        cfg = product_config(n_pages=16)
        R._META_CACHE.clear()
        R._meta_trained(recipe["seed"], pretrain_steps, cfg, meta_steps=recipe["meta_steps"])
        (state, loss), = R._META_CACHE.values()
    finally:
        for k, v in saved.items():
            setattr(R, k, v)
    return (registry or Registry()).save(model_id, cfg, state, recipe=recipe, loss=loss, note=note)


def toy_model(seed: int = 0, pretrain_steps: int = TOY_PRETRAIN):
    """The tiny CPU model `train("toy")` trains on, rebuilt identically (same seed, same steps)."""
    from ._train import meta as R
    saved = R.MODEL
    try:
        R.MODEL = None
        return R._fresh_lm(seed, pretrain_steps)
    finally:
        R.MODEL = saved
