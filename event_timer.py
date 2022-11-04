from __future__ import annotations
import heapq
import time
from typing import Callable, Any, List, Optional
from dataclasses import dataclass, field

Callback = Callable[[int], None]

@dataclass(frozen=True, order=True)
class Task:
    scheduled: int
    callback: Callback = field(compare=False)
    