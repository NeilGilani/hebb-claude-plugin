"""Where trained memories live, and how one is matched to a model.

A memory's read heads and key head are trained once per (base model, configuration). The
result is a few megabytes, kept in a registry folder with a JSON index, and found again by the
model id and a fingerprint of every configuration field that fixes a parameter shape.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from ._core.memory_lm import LMMemoryConfig

#: Configuration the library ships with. match_threshold above 1 means a new memory never
#: revises an existing page: one page per memory, which is what makes deletion and attribution
#: exact. n_pages is a capacity, not a shape, and is set per attach.
PRODUCT = dict(passes=20, inner_lr=1.0, match_threshold=1.01, cue_kind="attn")

#: The training recipe behind every real-model result in the paper: a key head trained against
#: hundreds of candidates and frozen, meta-training episodes shaped like deployment (one subject,
#: distinct attributes), pages written with the same range of inner steps deployment uses.
RECIPE = dict(key_steps=1500, key_slots="shared", freeze_keys=True, meta_episodes="keyed",
              meta_inner=(4, 20), meta_steps=800, seed=0)

#: Fields of LMMemoryConfig that change a trained parameter's shape or meaning.
SHAPE_FIELDS = ("page_size", "slot_dim", "key_dim", "heads", "read_every", "cue_kind")

HOME = Path(os.environ.get("HEBB_MEMORY_HOME") or Path.home() / ".cache" / "hebb-memory")


class NoCheckpoint(RuntimeError):
    """No trained memory exists for this model and configuration."""


class CheckpointMismatch(RuntimeError):
    """A checkpoint exists but was trained for a different model or configuration."""


def fingerprint(cfg: LMMemoryConfig) -> str:
    """Twelve hex characters that change when, and only when, a shape-fixing field changes."""
    shape = {k: getattr(cfg, k) for k in SHAPE_FIELDS}
    return hashlib.sha1(json.dumps(shape, sort_keys=True).encode()).hexdigest()[:12]


def product_config(n_pages: int = 256, **overrides) -> LMMemoryConfig:
    return LMMemoryConfig(n_pages=n_pages, **{**PRODUCT, **overrides})


def _safe(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", model_id).strip("-") or "model"


class Registry:
    """Trained memories on disk, one file per (model, fingerprint), with a JSON index."""

    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root is not None else HOME
        self.index_path = self.root / "index.json"

    def entries(self) -> List[Dict]:
        if not self.index_path.exists():
            return []
        return json.loads(self.index_path.read_text()).get("entries", [])

    def find(self, model_id: str, fp: str) -> Optional[Path]:
        for e in self.entries():
            if e["model"] == model_id and e["fingerprint"] == fp:
                p = self.root / e["file"]
                if p.exists():
                    return p
        return None

    def save(self, model_id: str, cfg: LMMemoryConfig, state: Dict, *, recipe: Dict,
             loss: float, note: str = "") -> Path:
        fp = fingerprint(cfg)
        self.root.mkdir(parents=True, exist_ok=True)
        fname = f"{_safe(model_id)}-{fp}.pt"
        path = self.root / fname
        payload = {"state": state, "loss": float(loss), "model": model_id, "fingerprint": fp,
                   "cfg": asdict(cfg), "recipe": dict(recipe), "note": note,
                   "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
        entries = [e for e in self.entries() if not (e["model"] == model_id and e["fingerprint"] == fp)]
        entries.append({"model": model_id, "fingerprint": fp, "file": fname, "loss": float(loss),
                        "created": payload["created"], "recipe": dict(recipe)})
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"entries": entries}, indent=1, default=str))
        os.replace(tmp, self.index_path)
        return path


def resolve(checkpoint: Optional[str | Path], model_id: str, fp: str, registry: Registry) -> Path:
    """A local path for the trained memory: an explicit file, a Hugging Face repo, or the registry.

    `checkpoint="hf:owner/repo"` downloads `memory.pt` from that model repo on the Hub.
    """
    if checkpoint is not None:
        spec = str(checkpoint)
        if spec.startswith("hf:"):
            from huggingface_hub import hf_hub_download
            repo, _, fname = spec[3:].partition("#")
            return Path(hf_hub_download(repo, fname or "memory.pt"))
        path = Path(spec).expanduser()
        if not path.exists():
            raise NoCheckpoint(f"checkpoint {path} does not exist")
        return path
    path = registry.find(model_id, fp)
    if path is None:
        raise NoCheckpoint(
            f"no trained memory for {model_id!r} (configuration {fp}) in {registry.root}.\n"
            f"Train one, once per base model:\n"
            f"    hebb-memory train --model {model_id}\n"
            f"or pass checkpoint='hf:<owner>/<repo>' to use one someone has published.")
    return path


def read_state(path: Path, model_id: str, fp: str) -> Tuple[Dict, Dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "state" not in payload:
        raise CheckpointMismatch(f"{path} is not a memory checkpoint (no 'state' entry)")
    got = payload.get("fingerprint")
    if got is not None and got != fp:
        raise CheckpointMismatch(
            f"{path} was trained for configuration {got}; this attach is {fp}. "
            f"The fields that must agree are {', '.join(SHAPE_FIELDS)}.")
    trained_on = payload.get("model")
    if trained_on is not None and trained_on != model_id:
        raise CheckpointMismatch(
            f"{path} was trained on {trained_on!r}, not {model_id!r}. A memory reads the hidden "
            f"states of the model it was trained on and means nothing on another one.")
    return payload["state"], payload
