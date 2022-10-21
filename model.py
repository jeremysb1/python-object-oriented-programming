from __future__ import annotations
from collections.abc import Iterator
import datetime
from typing import Optional, Union, Iterable

class Sample:
    def __init__(self,
                 sepal_length: float,
                 sepal_width: float,
                 petal_length: float,
                 petal_width: float,
                 species: Optional[float] = None,
    ) -> None:
        pass