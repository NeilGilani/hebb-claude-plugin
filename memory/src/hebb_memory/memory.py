"""Memory written into a frozen model's own forward pass, one page per memory.

    import hebb_memory as hebb
    mem = hebb.attach("Qwen/Qwen2.5-0.5B")                # needs a trained memory, see `train`
    r = mem.remember("acme", "refund window", "60 days")  # written into page r.page
    mem.ask("acme", "refund window")                      # the model answers, reading that page
    mem.trace("acme", "refund window")                    # which memories the answer read
    mem.forget(r)                                         # the page is zeroed and freed

The base model never changes. Every memory is written by gradient descent into its own small
page, and the model reads pages through trained read heads at every other layer. Because each
memory owns one page and nothing else changes when it is written, three things follow:

  * deletion is exact: forgetting a memory leaves the model exactly as if it had never been
    written (tests/test_memory.py checks this bit for bit);
  * attribution is exact: `trace` reports the pages the read actually used, with their weights;
  * memories are scoped: a question about one subject reads only that subject's pages, so a
    read across subjects (tenants, customers, users) is impossible rather than unlikely.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch

from ._core.data import Teaching
from ._core.memory_lm import MemoryLM
from .registry import (CheckpointMismatch, Registry, fingerprint, product_config, read_state,
                       resolve)

FORMAT = "hebb-memory/1"


class MemoryFull(RuntimeError):
    """Every page is in use. Forget something, or attach with more pages."""


@dataclass
class Record:
    """One memory and the page it lives in."""
    id: int
    subject: str
    attribute: str
    value: str
    page: int
    created: str
    write: Dict = field(default_factory=dict, repr=False)   # loss before/after, steps, change


@dataclass
class Hit:
    """A page a read used, and how much of the read it was."""
    record: Record
    weight: float
    cos: float


@dataclass
class Choice:
    best: str
    scores: Dict[str, float]          # total log-probability of each option


def question_for(subject: str, attribute: str) -> str:
    return f"Question: For {subject}, what is the {attribute}?\nAnswer:"


def teaching_for(subject: str, attribute: str, value: str, index: int,
                 rivals: Sequence[str] = ()) -> Teaching:
    """The shape every measured result was written from (a fact line plus two restatements)."""
    return Teaching(index=index, kind="fact", entity=subject, attribute=attribute, value=value,
                    text=f"For {subject}, the {attribute} is {value}.",
                    support=[(f"Note: for {subject}, {attribute}?", f" {value}"),
                             (f"{subject} {attribute} is", f" {value}")],
                    quizzes=[(question_for(subject, attribute), f" {value}")],
                    distractors=[v for v in rivals if v != value])


def _no_observe(z: torch.Tensor) -> None:
    return None


class Memory:
    """A frozen model with pages it can write, read, trace and delete. Made by `attach`."""

    def __init__(self, model: MemoryLM, info: Dict):
        self.model = model
        self.info = info
        # The research code keeps a running mean of every key it computes and centres
        # addresses on it. That makes every write and every question nudge shared state, so a
        # forgotten memory would leave a trace in the mean. Here the mean is the one fitted in
        # training and is never updated, which is what makes `forget` exact.
        model.keys.observe = _no_observe
        self._records: Dict[int, Record] = {}       # page -> record
        self._next_id = 0

    # ------------------------------------------------------------ writing
    def remember(self, subject: str, attribute: str, value: str, *,
                 rivals: Sequence[str] = ()) -> Record:
        """Write one memory into a fresh page. A second value for the same subject and
        attribute replaces the first: the old page is forgotten, not left to contradict it.

        `rivals` are other values this attribute could take. They are optional; when the
        configuration enables it, the write ranks the value above them.
        """
        subject, attribute, value = subject.strip(), attribute.strip(), value.strip()
        if not (subject and attribute and value):
            raise ValueError("subject, attribute and value must all be non-empty")
        old = self._find(subject, attribute)
        if not old and not bool((~self.model.allocated).any()):
            raise MemoryFull(f"all {self.capacity} pages are in use; forget something or attach "
                             f"with n_pages larger than {self.capacity}")
        for r in old:
            self._free(r.page)
        out = self.model.experience(teaching_for(subject, attribute, value, self._next_id, rivals))
        rec = Record(self._next_id, subject, attribute, value, int(out["page"]),
                     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     {"loss_start": out["loss_start"], "loss_end": out["loss_end"],
                      "steps": out["steps"], "change": out["state_change_l2"]})
        self._next_id += 1
        self._records[rec.page] = rec
        return rec

    # ------------------------------------------------------------ reading
    def ask(self, subject: str, attribute: str, max_new_tokens: int = 16) -> str:
        """The model's answer, reading only `subject`'s pages."""
        return self.generate(question_for(subject, attribute), subject=subject,
                             max_new_tokens=max_new_tokens)

    @torch.no_grad()
    def generate(self, prompt: str, *, subject: Optional[str] = None, max_new_tokens: int = 16,
                 stop_on_newline: bool = True) -> str:
        """Greedy decoding with memory attached. `subject` limits the read to its pages;
        None reads every page."""
        m = self.model
        m.eval()
        mem, _ = m._mem_for([prompt], restrict=self._restrict(subject))
        ids = m.tok(prompt, add_special_tokens=False).input_ids[-m.cfg.max_len:]
        ids = torch.tensor([ids], device=m.device)
        out: List[int] = []
        for _ in range(max_new_tokens):
            logits = m._forward(ids, torch.ones_like(ids), mem).logits[0, -1]
            nxt = int(logits.argmax())
            if nxt == m.tok.eos_token_id:
                break
            if stop_on_newline and "\n" in m.tok.decode([nxt]):
                break
            out.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=m.device)], dim=1)[:, -m.cfg.max_len:]
        return m.tok.decode(out).strip()

    @torch.no_grad()
    def choose(self, subject: str, attribute: str, options: Sequence[str]) -> Choice:
        """Which of `options` the model finds most likely, reading `subject`'s pages. This is
        the forced-choice score the paper's accuracy numbers are measured with."""
        m = self.model
        m.eval()
        q = question_for(subject, attribute)
        ids, mask, labels = m.encode_pairs([(q, f" {o}") for o in options])
        mem, _ = m._mem_for([q], restrict=self._restrict(subject))
        if mem is not None:
            mem = mem.expand(len(options), -1, -1)
        scores = m._seq_logprob(m._forward(ids, mask, mem).logits, labels).tolist()
        best = max(range(len(options)), key=lambda i: scores[i])
        return Choice(options[best], {o: round(float(s), 4) for o, s in zip(options, scores)})

    @torch.no_grad()
    def trace(self, subject: str, attribute: str) -> List[Hit]:
        """The pages a question about `subject`'s `attribute` reads, strongest first.

        This is the read path itself, not a reconstruction: the same addresses, the same
        top-k and the same softmax weights `ask` and `choose` use."""
        m = self.model
        pages = self._restrict(subject)
        if pages is None:
            return []
        q = m.keys.address(m.keys_for([question_for(subject, attribute)]))
        p = m.keys.address(m.K[pages])
        cos = (q @ p.T)[0]
        top, idx = cos.topk(min(m.cfg.read_k, len(pages)))
        w = (top / m.cfg.read_temp).softmax(-1)
        return [Hit(self._records[int(pages[i])], round(float(wt), 4), round(float(c), 4))
                for i, wt, c in zip(idx.tolist(), w, top)]

    # ------------------------------------------------------------ deleting
    def forget(self, target: Union[Record, int, str]) -> List[Record]:
        """Delete a memory (a Record or its id), or every memory of a subject (a string).

        The page is zeroed and freed. Nothing else in the model was changed by writing it, so
        nothing else needs undoing: what remains is exactly the model without that memory.
        """
        if isinstance(target, Record):
            gone = [r for r in self._records.values() if r.id == target.id]
        elif isinstance(target, int):
            gone = [r for r in self._records.values() if r.id == target]
        elif isinstance(target, str):
            gone = [r for r in self._records.values() if r.subject == target]
        else:
            raise TypeError("forget takes a Record, a record id, or a subject")
        for r in gone:
            self._free(r.page)
        return gone

    def _free(self, page: int) -> None:
        m = self.model
        for b in ("M", "K", "usage", "importance", "last_delta", "write_count", "read_count"):
            getattr(m, b)[page] = 0
        m.allocated[page] = False
        m.owner[page] = -1
        self._records.pop(page, None)

    # ------------------------------------------------------------ inspecting
    def records(self, subject: Optional[str] = None) -> List[Record]:
        rs = sorted(self._records.values(), key=lambda r: r.id)
        return [r for r in rs if subject is None or r.subject == subject]

    def subjects(self) -> List[str]:
        return sorted({r.subject for r in self._records.values()})

    @property
    def capacity(self) -> int:
        return int(self.model.cfg.n_pages)

    def bytes_per_memory(self) -> int:
        """Stored floats per memory: its page and its key, at four bytes each."""
        c = self.model.cfg
        return (c.page_size * c.slot_dim + c.key_dim) * 4

    def _find(self, subject: str, attribute: str) -> List[Record]:
        return [r for r in self._records.values() if r.subject == subject and r.attribute == attribute]

    def _restrict(self, subject: Optional[str]) -> Optional[torch.Tensor]:
        if subject is None:
            pages = sorted(self._records)
        else:
            pages = sorted(p for p, r in self._records.items() if r.subject == subject)
        if not pages:
            return None
        return torch.tensor(pages, dtype=torch.long, device=self.model.device)

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return (f"Memory(model={self.info.get('model')!r}, memories={len(self)}, "
                f"capacity={self.capacity})")

    # ------------------------------------------------------------ saving
    def save(self, path: Union[str, Path], subject: Optional[str] = None) -> Path:
        """Write the memories (all of them, or one subject's) to a file. Pages only: a few
        kilobytes per memory, not a copy of the model."""
        keep = self.records(subject)
        m = self.model
        payload = {"format": FORMAT, "model": self.info.get("model"),
                   "fingerprint": self.info.get("fingerprint"),
                   "records": [{k: v for k, v in asdict(r).items()} for r in keep],
                   "M": m.M[[r.page for r in keep]].detach().cpu(),
                   "K": m.K[[r.page for r in keep]].detach().cpu()}
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        return path

    def load(self, path: Union[str, Path]) -> List[Record]:
        """Add the memories saved in `path`. A memory for a subject and attribute that already
        exists here replaces it, as `remember` would."""
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != FORMAT:
            raise CheckpointMismatch(f"{path} is not a saved hebb-memory file")
        for k in ("model", "fingerprint"):
            if payload[k] != self.info.get(k):
                raise CheckpointMismatch(f"{path} was saved from {k} {payload[k]!r}; this memory "
                                         f"is {self.info.get(k)!r}")
        m = self.model
        loaded = []
        for i, d in enumerate(payload["records"]):
            for r in self._find(d["subject"], d["attribute"]):
                self._free(r.page)
            free = (~m.allocated).nonzero().flatten()
            if not len(free):
                raise MemoryFull(f"no free page for {d['subject']}/{d['attribute']}")
            page = int(free[0])
            m.M[page] = payload["M"][i].to(m.M.device)
            m.K[page] = payload["K"][i].to(m.K.device)
            m.allocated[page] = True
            m.usage[page] = 1.0
            rec = Record(**{**d, "id": self._next_id, "page": page})
            m.owner[page] = rec.id
            self._next_id += 1
            self._records[page] = rec
            loaded.append(rec)
        return loaded


# ---------------------------------------------------------------- attach

def _load_model(model_id: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    lm = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    return lm, tok


def attach(model, tokenizer=None, *, model_id: Optional[str] = None,
           checkpoint: Optional[Union[str, Path]] = None, registry: Optional[Registry] = None,
           device: Optional[str] = None, n_pages: int = 256, **cfg_overrides) -> Memory:
    """A frozen model with a trained memory on it.

    `model` is a Hugging Face model id, or an already-loaded causal LM (then pass `tokenizer`).
    The trained memory comes from `checkpoint` (a file, or "hf:owner/repo" on the Hugging Face
    Hub), else from the local registry that `hebb-memory train` fills. Nothing here trains.
    """
    if isinstance(model, str):
        model_id = model
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        lm, tok = _load_model(model_id, device)
    else:
        if tokenizer is None:
            raise ValueError("pass the tokenizer with a loaded model")
        lm, tok = model, tokenizer
        model_id = model_id or getattr(getattr(lm, "config", None), "_name_or_path", None) \
            or lm.__class__.__name__
        device = device or str(next(lm.parameters()).device)
    cfg = product_config(n_pages=n_pages, **cfg_overrides)
    fp = fingerprint(cfg)
    path = resolve(checkpoint, model_id, fp, registry or Registry())
    state, payload = read_state(path, model_id, fp)
    m = MemoryLM(lm, tok, cfg).to(device)
    m.load_checkpoint(state)
    if m.read_heads_are_inert():
        raise CheckpointMismatch(f"{path} loaded but its read heads are zero: it holds no "
                                 f"trained memory")
    m.begin_stream()
    return Memory(m, {"model": model_id, "fingerprint": fp, "checkpoint": str(path),
                      "recipe": payload.get("recipe"), "loss": payload.get("loss")})


def selfcheck(mem: Memory, attribute: str = "refund window",
              facts: Iterable[Tuple[str, str]] = (("acme", "60 days"), ("globex", "14 days"))) -> Dict:
    """Is the memory attached and working mechanically? Two subjects, one attribute they
    disagree on. Leaves the memory as it found it.

    Checks: the read heads are trained, each write changes its page, a question routes to its
    own subject's page when every page is a candidate, and forgetting empties the page.
    Accuracy is reported but does not gate it: a small model can answer wrong with everything
    attached right, and that is a research number, not a plumbing one."""
    if len(mem):
        raise RuntimeError("selfcheck needs an empty memory")
    facts = list(facts)
    values = [v for _, v in facts]
    m = mem.model
    rep: Dict = {"read_heads_active": not m.read_heads_are_inert(), "writes": [], "routing": [],
                 "answers": []}
    recs = [mem.remember(s, attribute, v, rivals=values) for s, v in facts]
    for r in recs:
        rep["writes"].append({"subject": r.subject, "page": r.page, "change": round(r.write["change"], 4)})
    rep["writes_changed_pages"] = all(w["change"] > 0 for w in rep["writes"])
    correct = 0
    for r in recs:
        _, idx = m._mem_for([question_for(r.subject, attribute)], restrict=mem._restrict(None))
        routed = int(idx.flatten()[0])
        rep["routing"].append({"subject": r.subject, "routed_to": routed, "own": r.page})
        got = mem.choose(r.subject, attribute, values).best
        correct += got == r.value
        rep["answers"].append({"subject": r.subject, "want": r.value, "got": got})
    rep["routing_ok"] = all(x["routed_to"] == x["own"] for x in rep["routing"])
    rep["forced_choice_accuracy"] = correct / len(facts)
    for r in recs:
        mem.forget(r)
    rep["forget_ok"] = (not bool(m.allocated.any())) and float(m.M.abs().sum()) == 0.0
    rep["ok"] = bool(rep["read_heads_active"] and rep["writes_changed_pages"]
                     and rep["routing_ok"] and rep["forget_ok"])
    return rep
