from __future__ import annotations
from asyncio import SubprocessTransport
from collections.abc import Iterator
import abc
import csv
import datetime
import enum
from importlib.resources import path
from pathlib import Path
import random
from typing import cast, overload, Any, Optional, Union, Iterable, Iterator, List, Dict, Counter, Callable, Protocol, TypedDict, TypeVar, DefaultDict, overload
import weakref

class InvalidSampleError(ValueError):
    """Source data has invalid data representation."""

class Sample:
    def __init__(
        self,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
        species: Optional[float] = None,
    ) -> None:
        self.sepal_length = sepal_length
        self.sepal_width = sepal_width
        self.petal_length = petal_length
        self.petal_width = petal_width
        self.species = species
        self.classification: Optional[str] = None
    
    def __repr__(self) -> str:
        return(
            f"{self.__class__.__name__}("
            f"sepal_length={self.sepal_length}, "
            f"sepal_width={self.sepal_width}, "
            f"petal_length={self.petal_length}, "
            f"petal_width={self.petal_width}, "
            f")"
        )

class Purpose(enum.IntEnum):
    Classification = 0
    Testing = 1
    Training = 2

class KnownSample(Sample):
    """Represents a sample of testing or training data, the species is set once
    The purpose determines if it can or cannot be classified.
    """

    def __init__(
        self,
        species: str,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
        purpose: int,
    ) -> None:
        purpose_enum = Purpose(purpose)
        if purpose_enum not in {Purpose.Training, Purpose.Testing}:
            raise ValueError(f"Invalid: purpose: {purpose!r}: {purpose_enum}")
        super().__init__(
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width,
        )
        self.purpose = purpose_enum
        self.species = species
        self._classification: Optional[str] = None

    def matches(self) -> bool:
        return self.species == self.classification
    
    @property
    def classification(self) -> Optional[str]:
        if self.purpose == Purpose.Testing:
            return self._classification
        else:
            raise AttributeError(f"Training samples have no classification.")
    
    @classification.setter
    def classification(self, value: str) -> None:
        if self.purpose == Purpose.Testing:
            self._classification = value
        else:
            raise AttributeError(f"Training samples can't be classified.")
    
    def __repr__(self) -> str:
        base_attributes = self.attr_dict
        base_attributes["purpose"] = f"{self.purpose.value}"
        base_attributes["species"] = f"{self.species!r}"
        if self.purpose == Purpose.Testing and self._classification:
            base_attributes["classification"] = f"{self._classification!r}"
        attrs = ", ".join(f"{k}={v}" for k, v in base_attributes.items())
        return f"{self.__class__.__name__}({attrs})"

class TrainingKnownSample(KnownSample):
    """Training data."""

    @classmethod
    def from_dict(cls, row: dict[str, str]) -> "TrainingKnownSample":
        return cast(TrainingKnownSample, super().from_dict(row))

class TestingKnownSample(KnownSample):
    """Testing data. A classifier can assign a species, which may or may not be correct."""

    def __init__(
        self, 
        species: str, 
        sepal_length: float, 
        sepal_width: float, 
        petal_length: float, 
        petal_width: float,
        classification: Optional[str] = None,
    ) -> None:
        super().__init__(
            species=species, 
            sepal_length=sepal_length, 
            sepal_width=sepal_width, 
            petal_length=petal_length, 
            petal_width=petal_width,
        )
        self.classification = classification

    def matches(self) -> bool:
        return self.species == self.classification
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"sepal_length={self.sepal_length}, "
            f"sepal_width={self.sepal_width}, "
            f"petal_length={self.petal_length}, "
            f"petal_width={self.petal_width}, "
            f"species={self.species!r}, "
            f"classification={self.classification!r}, "
            f")"
        )
    
    @classmethod
    def from_dict(cls, row: dict[str, str]) -> "TestingKnownSample":
        return cast(TestingKnownSample, super().from_dict(row))

class UnknownSample(Sample):
    """A sample not yet classified."""

    @classmethod
    def from_dict(cls, row: dict[str, str]) -> "UnknownSample":
        if set(row.keys()) != {
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        }:
            raise InvalidSampleError(f"invalid fields in {row!r}")
        try:
            return cls(
                sepal_length=float(row["sepal_length"]),
                sepal_width=float(row["sepal_width"]),
                petal_length=float(row["petal_length"]),
                petal_width=float(row["petal_width"]),
            )
        except (ValueError, KeyError) as ex:
            raise InvalidSampleError(f"invalid {row!r}")

class Hyperparameter:
    """A hyperparameter value and the overall quality of the classification."""
    def __init__(self, k: int, training: "TrainingData") -> None:
        self.k = k
        self.data: weakref.ReferenceType["TrainingData"] = weakref.ref(training)
        self.quality = float
    
    def test(self) -> None:
        """Run the entire test suite."""
        training_data: Optional["TrainingData"] = self.data()
        if not training_data:
            raise RuntimeError("Broken Weak Reference")
        pass_count, fail_count = 0, 0
        for sample in training_data.testing:
            sample.classification = self.classify(sample)
            if sample.matches():
                pass_count += 1
            else:
                fail_count += 1
        self.quality = pass_count / (pass_count + fail_count)

class SampleDict(TypedDict):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    species: str

class SamplePartition(List[SampleDict], abc.ABC):
    @overload
    def __init__(self, *, training_subset: float = 0.80) -> None:
        ...

    @overload
    def __init__(self, iterable: Optional[Iterable[SampleDict]] = None, *, training_subset: float = 0.80) -> None:
        ...

    def __init__(self, iterable: Optional[Iterable[SampleDict]] = None, *, training_subset: float = 0.80) -> None:
        self.training_subset = training_subset
        if iterable:
            super().__init__(iterable)
        else:
            super().__init__()
    
    @abc.abstractproperty
    @property
    def training(self) -> List[TrainingKnownSample]:
        ...
    
    @abc.abstractproperty
    @property
    def testing(self) -> List[TestingKnownSample]:
        ...

class ShufflingSamplePartition(SamplePartition):
    def __init__(self, iterable: Optional[Iterable[SampleDict]] = None, *, training_subset: float = 0.80) -> None:
        super().__init__(iterable, training_subset=training_subset)
        self.split: Optional[int] = None
    
    def shuffle(self) -> None:
        if not self.split:
            random.shuffle(self)
            self.split = int(len(self) * self.training_subset)
    
    @property
    def training(self) -> List[TrainingKnownSample]:
        self.shuffle()
        return [TrainingKnownSample(**sd) for sd in self[: self.split]]
    
    @property
    def testing(self) -> List[TestingKnownSample]:
        self.shuffle()
        return [TrainingKnownSample(**sd) for sd in self[self.split :]]

class DealingPartition(abc.ABC):
    @abc.abstractmethod
    def __init__(self, items: Optional[Iterable[SampleDict]], *, training_subset: Tuple[int, int] = (8, 10),) -> None:
        ...
    
    @abc.abstractmethod
    def extend(self, items: Iterable[SampleDict]) -> None:
        ...
    
    @abc.abstractmethod
    def append(self, item: SampleDict) -> None:
        ...
    
    @property
    @abc.abstractmethod
    def training(self) -> List[TrainingKnownSample]:
        ...
    
    @property
    @abc.abstractmethod
    def testing(self) -> List[TestingKnownSample]:
        ...

class TrainingData:
    """A set of training and testing data with methods to load and test the samples."""
    def __init__(self, name: str) -> None:
        self.name = name
        self.uploaded: datetime.datetime
        self.tested: datetime.datetime
        self.training: list[Sample] = []
        self.testing: list[Sample] = []
        self.tuning: list[Hyperparameter] = []
    
    def load(self, raw_data_source: Iterable[dict[str, str]]) -> None:
        """Load and partition the raw data."""
        for n, row in enumerate(raw_data_source):
            ... filter and extract subsets 
            ... Create self.training and self.testing subsets 
    
    def test(self, parameter: Hyperparameter) -> None:
        """Test the parameter value."""
        parameter.test()
        self.tuning.append(parameter)
        self.tested = datetime.datetime.now(tz=datetime.timezone.utc)
    
    def classify(self, parameter: Hyperparameter, sample: Sample) -> Sample:
        """Classify this sample."""
        classification = parameter.classify(sample)
        sample.classify(classification)
        return sample

class BadSampleRow(ValueError):
    pass

class SampleReader:
    """See iris.names for attribute ordering in bezdekIris.data file"""

    target_class = Sample
    header = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

    def __init__(self, source: Path) -> None:
        self.source = source
    
    def sample_iter(self) -> Iterator[Sample]:
        target_class = self.target_class
        with self.source.open() as source_file:
            reader = csv.DictReader(source_file, self.header)
            for row in reader:
                try:
                    sample = target_class(
                        sepal_length=float(row["sepal_length"]),
                        sepal_width=float(row["sepal_width"]),
                        petal_length=float(row["petal_length"]),
                        petal_width=float(row["petal_width"])
                    )
                except ValueError as ex:
                    raise BadSampleRow(f"Invalid {row!r}") from ex
                yield Sample

class Distance:
    """Definition of a distance calculation."""

    def distance(self, s1: Sample, s2: Sample) -> float:
        pass

class ED(Distance):
    def distance(self, s1: Sample, s2: Sample) -> float:
        return hypot(
            s1.sepal_length - s2.sepal_length,
            s1.sepal_width - s2.sepal_width,
            s1.petal_length - s2.petal_length,
            s1.petal_width - s2.petal_width
        )

class MD(Distance):
    def distance(self, s1: Sample, s2: Sample) -> float:
        return sum(
            [
                abs(s1.sepal_length - s2.sepal_length),
                abs(s1.sepal_width - s2.sepal_width),
                abs(s1.petal_length - s2.petal_length),
                abs(s1.petal_width - s2.petal_width),
            ]
        )

class CD(Distance):
    def distance(self, s1: Sample, s2: Sample) -> float:
        return max(
            [
                abs(s1.sepal_length - s2.sepal_length),
                abs(s1.sepal_width - s2.sepal_width),
                abs(s1.petal_length - s2.petal_length),
                abs(s1.petal_width - s2.petal_width),
            ]
        )

class SD(Distance):
    def distance(self, s1: Sample, s2: Sample) -> float:
        return sum(
            [
                abs(s1.sepal_length - s2.sepal_length),
                abs(s1.sepal_width - s2.sepal_width),
                abs(s1.petal_length - s2.petal_length),
                abs(s1.petal_width - s2.petal_width),
            ]
        ) / sum(
            [
                s1.sepal_length + s2.sepal_length,
                s1.sepal_width + s2.sepal_width,
                s1.petal_length + s2.petal_length,
                s1.petal_width + s2.petal_width,
            ]
        )