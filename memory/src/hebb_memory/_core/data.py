"""Teach → quiz benchmark.

A *teaching* is one thing a user tells the assistant.  It comes with
  support: the teaching sentence + a few QA restatements (what the model learns from)
  quizzes: held-out questions with short answers (what it is scored on)
Some quizzes require TWO teachings about the same entity ("combined"); they exist so that
pure recall of a single stored sentence is not enough.

Entities are split disjointly between the pretraining and stream distributions.
Everything is deterministic given (split, index, seed).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

NAMES_TRAIN = ["Ada", "Bruno", "Cleo", "Dev", "Esme", "Farid", "Gia", "Hugo", "Ines", "Jonah", "Kira", "Leon",
               "Mira", "Nico", "Ola", "Pia", "Quinn", "Rafa", "Sol", "Theo", "Uma", "Vik", "Wren", "Xavi", "Yara", "Zed"]
NAMES_EVAL = ["Amara", "Bodhi", "Cyrus", "Dalia", "Elio", "Freya", "Gus", "Hana", "Ivo", "Juno", "Kai", "Lior",
              "Maeve", "Nadia", "Oskar", "Priya", "Remy", "Sana", "Tomas", "Ulla", "Vera", "Wes", "Ximena", "Yusuf", "Zora"]
ATTRIBUTES: Dict[str, List[str]] = {
    "favourite colour": ["red", "blue", "green", "yellow", "purple", "orange", "black", "white", "teal", "pink"],
    "home city": ["Lisbon", "Oslo", "Denver", "Kyoto", "Lima", "Cairo", "Perth", "Quito", "Riga", "Seoul"],
    "pet": ["a dog", "a cat", "a parrot", "a rabbit", "a turtle", "a hamster", "a goat", "a snake"],
    "lucky number": ["3", "7", "11", "12", "19", "23", "42", "64", "77", "99"],
    "job": ["a nurse", "a pilot", "a baker", "a coder", "a farmer", "a judge", "a diver", "a tailor"],
    "favourite food": ["sushi", "tacos", "pasta", "curry", "pizza", "ramen", "falafel", "stew"],
}
RULES: List[Tuple[str, str]] = [
    ("ping", "pong"), ("status", "all systems nominal"), ("go", "green light"), ("code word", "falcon"),
    ("sign off", "over and out"), ("wake word", "sesame"), ("password hint", "the river"), ("motto", "ship it"),
    ("greeting", "howdy partner"), ("alarm word", "red sky"), ("safe word", "pineapple"), ("checkpoint", "alpha bravo"),
]


@dataclass
class Teaching:
    index: int
    kind: str                       # "fact" | "rule"
    entity: str
    attribute: str
    value: str
    text: str                       # the teaching utterance (also the cue source)
    support: List[Tuple[str, str]]  # (prompt, answer) pairs the model may learn from
    quizzes: List[Tuple[str, str]]  # held-out (prompt, answer)
    combined_with: Optional[int] = None   # index of an earlier teaching a combined quiz depends on
    combined_quiz: Optional[Tuple[str, str]] = None
    meta: Dict = field(default_factory=dict)
    distractors: List[str] = field(default_factory=list)  # rival values for the same attribute, if
                                                          # the source knows them (tenant rules do);
                                                          # what forced-choice scoring compares against


def _rng(split: str, idx: int, seed: int) -> np.random.Generator:
    h = hashlib.sha256(f"teach|{split}|{idx}|{seed}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def teaching_from_text(text: str, index: int = 0) -> Teaching:
    """Wrap a plain sentence as a Teaching.

    The product surface is `model.experience("...")`: one sentence, no schema. There is no
    held-out quiz for such a teaching, so `quizzes` is empty; the support set is the
    sentence itself, phrased two ways so the write is not fitted to a single surface form.
    """
    text = text.strip()
    return Teaching(
        index=index, kind="freeform", entity="", attribute="", value=text, text=text,
        support=[("Fact: ", text), (f"Repeat the note.\nNote:", " " + text)],
        quizzes=[], meta={"source": "freeform"},
    )


def restate(t: "Teaching") -> str:
    """The same fact said differently. Should land on the SAME page as `t`."""
    if t.kind == "rule":
        return f"Reminder: {t.entity} means you answer {t.value}."
    return f"By the way, {t.entity} has {t.attribute} {t.value}."


def revise(t: "Teaching", new_value: str) -> "Teaching":
    """The same slot of knowledge with a NEW value. Should also land on `t`'s page, and
    overwrite it -- this is the update path a user exercises constantly ("actually, it
    moved"). Splitting it onto a fresh page leaves two contradictory memories."""
    if t.kind == "rule":
        text = f"Rule: when I say {t.entity}, reply with {new_value}."
        support = [(f"When I say {t.entity} you say:", f" {new_value}"), (f"Rule {t.entity} ->", f" {new_value}")]
        quizzes = [(f"User: {t.entity}\nAssistant:", f" {new_value}")]
    else:
        text = f"Actually, {t.entity}'s {t.attribute} is now {new_value}."
        support = [(f"Note: {t.entity}'s {t.attribute}?", f" {new_value}"), (f"{t.entity}'s {t.attribute} is", f" {new_value}")]
        quizzes = [(f"Question: What is {t.entity}'s {t.attribute}?\nAnswer:", f" {new_value}")]
    return Teaching(t.index, t.kind, t.entity, t.attribute, new_value, text, support, quizzes,
                    meta={"revises": t.index})


BENCHMARK_VERSION = 2   # 1 = support and quiz shared a surface form (leaky); 2 = disjoint


def teaching_payload(t: "Teaching") -> List[str]:
    """The material a model is allowed to learn one teaching from.

    Every model in the Stage 1 comparison receives exactly this: the teaching sentence and
    its support restatements. Neural Memory writes it into a page; the context baseline puts
    it in the prompt; the retrieval baseline indexes it. Before 2026-09-09 the baselines got
    only `t.text` while the memory model got the restatements as well, which made them
    weakened baselines rather than real ones.
    """
    return [t.text] + [(p.replace("\n", " ").strip() + a) for p, a in t.support]


class TeachQuizGenerator:
    def __init__(self, seed: int = 0, combined_every: int = 3):
        self.seed = seed
        self.combined_every = combined_every

    def make(self, idx: int, split: str, prev: Optional[List["Teaching"]] = None) -> Teaching:
        rng = _rng(split, idx, self.seed)
        names = NAMES_TRAIN if split == "train" else NAMES_EVAL
        kind = "rule" if rng.random() < 0.25 else "fact"
        if kind == "fact":
            entity = names[int(rng.integers(len(names)))]
            attribute = list(ATTRIBUTES)[int(rng.integers(len(ATTRIBUTES)))]
            value = ATTRIBUTES[attribute][int(rng.integers(len(ATTRIBUTES[attribute])))]
            text = f"Remember: {entity}'s {attribute} is {value}."
            # Support and quiz surface forms are DISJOINT: no quiz prompt ever appears in the
            # support set. Until 2026-09-09 they shared their first item, which meant half of
            # every quiz was material the memory model had trained on. See RESULTS_STAGE1.md §3.
            support = [(f"Note: {entity}'s {attribute}?", f" {value}"),
                       (f"{entity}'s {attribute} is", f" {value}")]
            quizzes = [(f"Question: What is {entity}'s {attribute}?\nAnswer:", f" {value}"),
                       (f"Question: Tell me {entity}'s {attribute}.\nAnswer:", f" {value}")]
        else:
            trig, resp = RULES[int(rng.integers(len(RULES)))]
            entity, attribute, value = trig, "rule", resp
            text = f"Rule: when I say {trig}, reply with {resp}."
            support = [(f"When I say {trig} you say:", f" {resp}"), (f"Rule {trig} ->", f" {resp}")]
            quizzes = [(f"User: {trig}\nAssistant:", f" {resp}"),
                       (f"Question: What do you reply when I say {trig}?\nAnswer:", f" {resp}")]
        t = Teaching(idx, kind, entity, attribute, value, text, support, quizzes)
        # combined quiz: needs an earlier fact about the same entity
        if prev and kind == "fact":
            earlier = [p for p in prev if p.kind == "fact" and p.entity == entity and p.attribute != attribute]
            if earlier and idx % self.combined_every == 0:
                e = earlier[-1]
                t.combined_with = e.index
                t.combined_quiz = (f"Question: What are {entity}'s {e.attribute} and {attribute}?\nAnswer:", f" {e.value} and {value}")
        return t

    def stream(self, n: int, split: str = "eval") -> List[Teaching]:
        out: List[Teaching] = []
        for i in range(n):
            out.append(self.make(i, split, out))
        return out

    def episode(self, n: int, rng: np.random.Generator, split: str = "train") -> List[Teaching]:
        out: List[Teaching] = []
        base = int(rng.integers(0, 10**8))
        for i in range(n):
            t = self.make(base + i, split, out)
            t.index = i
            out.append(t)
        return out

    def corpus(self, n: int = 400) -> List[str]:
        """Text for training a toy tokenizer."""
        rng = np.random.default_rng(self.seed)
        lines = []
        for i in range(n):
            t = self.make(i, "train" if i % 2 else "eval")
            lines += [t.text] + [p + a for p, a in t.support + t.quizzes]
            if t.combined_quiz:
                lines.append(t.combined_quiz[0] + t.combined_quiz[1])
        lines += [f"{n}'s {a} is {v}." for a, vs in ATTRIBUTES.items() for v in vs for n in NAMES_TRAIN + NAMES_EVAL]
        lines += [f"{a} and {b}" for a in sum(ATTRIBUTES.values(), []) for b in sum(ATTRIBUTES.values(), [])[:10]]
        return lines
