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

@dataclass
class KnownSample(Sample):
    species: str

@dataclass
class TestingKnownSample(KnownSample):
    classification: Optional[str] = None

@dataclass
class TrainingKnownSample(KnownSample):
    """Note: no classification instance variable available."""
    pass

@dataclass
class Hyperparameter:
    """A specific tuning parameter set with k and a distance algorithm"""
    k: int
    algorithm: Distance
    data: weakref.ReferenceType["TrainingData"]
    def classify(self, sample: Sample) -> str:
        """The k-NN algorithm"""
        ...