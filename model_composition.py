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

class TestingKnownSample:
    def __init__(self, sample: KnownSample, classification: Optional[str] = None) -> None:
        self.sample = sample
        self.classification = classification
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(sample={self.sample!r}," 
            f"classification={self.classification!r})"
        )
    
class TrainingKnownSample(NamedTuple):
    sample: KnownSample