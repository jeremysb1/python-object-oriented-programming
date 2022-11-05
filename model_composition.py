from __future__ import annotations
import bisect
import heapq
import collections
from typing import cast, NamedTuple, Callable, Iterable, List, Union, Counter


class Sample(NamedTuple):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class KnownSample(NamedTuple):
    sample: Sample
    species: str

class TestingKnownSample(NamedTuple):
    sample: KnownSample
   
class TrainingKnownSample(NamedTuple):
    sample: KnownSample

TrainingList = List[TrainingKnownSample]
TestingList = List[TestingKnownSample]

class UnknownSample(NamedTuple):
    sample: Sample

class ClassifiedKnownSample(NamedTuple):
    sample: KnownSample
    classification: str

class ClassifiedUnknownSample(NamedTuple):
    sample: UnknownSample
    classification: str

AnySample = Union[KnownSample, UnknownSample]
DistanceFunc = Callable[[TrainingKnownSample, AnySample], float]

class Measured(NamedTuple):
    """Measured distance is first to simplify sorting."""

    distance: float
    sample: TrainingKnownSample

import itertools
from typing import DefaultDict[int, List[KnownSample]]

Classifier = Callable[[int, DistanceFunc, TrainingList, AnySample], str]

