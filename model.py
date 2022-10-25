from __future__ import annotations
from asyncio import SubprocessTransport
from collections.abc import Iterator
import datetime
from typing import cast, Optional, Union, Iterable
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

class KnownSample(Sample):
    """Abstract superclass for testing data, the species is set externally."""

    def __init__(
        self,
        species: str,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
    ) -> None:
        super().__init__(
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width,
        )
        self.species = species
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"sepal_length={self.sepal_length}, "
            f"sepal_width={self.sepal_width}, "
            f"petal_length={self.petal_length}, "
            f"petal_width={self.petal_width}, "
            f"species={self.species!r}"
            f")"
        )
    
    @classmethod
    def from_dict(cls, row: dict[str, str]) -> "KnownSample":
        if row["species"] not in {"Iris-setosa", "Iris-versicolour", "Iris-virginica"}:
            raise InvalidSampleError(f"invalid species in {row!r}")
        try:
            return cls(
                species=row["species"],
                sepal_length=float(row["sepal_length"]),
                sepal_width=float(row["sepal_width"]),
                petal_length=float(row["petal_length"]),
                petal_width=float(row["petal_width"]),
            )
        except ValueError as ex:
            raise InvalidSampleError(f"invalid {row!r}")

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

class TrainingData:
    def __init__(self, name: str) -> None:
        self.name = name
        self.uploaded: datetime.datetime
        self.tested: datetime.datetime
        self.training: list[TrainingKnownSample] = []
        self.testing: list[TestingKnownSample] = []
        self.tuning: list[Hyperparameter] = []

    def load(self, raw_data_iter: Iterable[dict[str, str]]) -> None:
        for n, row in enumerate(raw_data_iter):
            try:
                if n % 5 == 0:
                    test = TestingKnownSample.from_dict(row)
                    self.testing.append(test)
                else:
                    train = TrainingKnownSample.from_dict(row)
                    self.training.append(train)
            except InvalidSampleError as ex:
                print(f"Row {n+1}: {ex}")
                return
        self.uploaded = datetime.datetime.now(tz=datetime.timezone.utc)