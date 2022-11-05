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

class Hyperparameter(NamedTuple):
    k: int
    distance_function: DistanceFunc
    training_data: TrainingList
    classifier: Classifier
    
    def classify(self, unknown: AnySample) -> str:
        classifier: Classifier
        return classifier(self.k, self.distance_function, self.training_data, unknown)
    
    def test(self, testing: TestingList) -> int:
        classifier: self.classifier
        test_results = (
            ClassifiedKnownSample(
                t.sample, 
                classifier(
                    self.k, self.distance_function, self.training_data, t.sample
                ),
            )
            for t in testing
        )
        pass_fail = map(
            lambda t: (1 if t.sample.species == t.classification else 0), test_results
        )
        return sum(pass_fail)

def minkowski(
    s1: TrainingKnownSample,
    s2: AnySample,
    m: int,
    summarize: Callable[[Iterable[float]], float] = sum,
) -> float:
    return (
        summarize(
            [
                abs(s1.sample.sample.sepal_length - s2.sample.sepal_length) ** m,
                abs(s1.sample.sample.sepal_width - s2.sample.sepal_width) ** m,
                abs(s1.sample.sample.petal_length - s2.sample.petal_length) ** m,
                abs(s1.sample.sample.petal_width - s2.sample.petal_width) ** m,
            ]
        )
        ** (1 / m)
    )

def manhattan(s1: TrainingKnownSample, s2: AnySample) -> float:
    return minkowski(s1, s2, m=1)

def euclidean(s1: TrainingKnownSample, s2: AnySample) -> float:
    return minkowski(s1, s2, m=2)

def chebyshev(s1: TrainingKnownSample, s2: AnySample) -> float:
    return minkowski(s1, s2, m=1, summarize=max)