from __future__ import annotations
import collections
from dataclasses import dataclass, asdict
from typing import Optional, Counter, List
import weakref
import sys

@dataclass
class Sample:
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    