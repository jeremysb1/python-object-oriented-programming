from __future__ import annotations
from typing import (
    cast,
    Callable,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Type,
    Union,
)
from collections import defaultdict, Counter
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

class TestingKnownSample(NamedTuple):
    sample: KnownSample
   
class TrainingKnownSample(NamedTuple):
    sample: KnownSample

def training_80(s: KnownSample, i: int) -> bool:
    return i % 5 != 0

def training_75(s: KnownSample, i: int) -> bool:
    return i % 4 != 0

def training_67(s: KnownSample, i: int) -> bool:
    return i % 3 != 0

TrainingList = List[TrainingKnownSample]
TestingList = List[TestingKnownSample]

def partition(
    samples: Iterable[KnownSample],
    rule: Callable[[KnownSample, int], bool]
) -> tuple[TrainingList, TestingList]:
    training_samples = [
        TrainingKnownSample(s) for i, s in enumerate(samples) if rule(s, i)
    ]
    test_samples = [
        TestingKnownSample(s) for i, s in enumerate(samples) if rule(s, i)
    ]

    return training_samples, test_samples