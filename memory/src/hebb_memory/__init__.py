"""hebb-memory: memory written into a frozen model, one page per memory.

    import hebb_memory as hebb
    mem = hebb.attach("Qwen/Qwen2.5-0.5B")
    r = mem.remember("acme", "refund window", "60 days")
    mem.ask("acme", "refund window")
    mem.trace("acme", "refund window")
    mem.forget(r)
"""
from .memory import Choice, Hit, Memory, MemoryFull, Record, attach, selfcheck
from .registry import CheckpointMismatch, NoCheckpoint, Registry
from .training import train

__version__ = "0.1.0"
__all__ = ["attach", "train", "selfcheck", "Memory", "Record", "Hit", "Choice", "Registry",
           "MemoryFull", "NoCheckpoint", "CheckpointMismatch", "__version__"]
