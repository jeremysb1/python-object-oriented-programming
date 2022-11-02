import collections
from typing import Optional, Counter, NamedTuple
import weakref
import sys

class Sample(NamedTuple):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class KnownSample(NamedTuple):
    sample: Sample
    species: str
