from __future__ import annotations
import heapq
import time
from typing import Callable, Any, List, Optional
from dataclasses import dataclass, field

Callback = Callable[[int], None]

@dataclass(frozen=True, order=True)
class Task:
    """
    Definition of a task to be performed at an interval after the Scheduler starts.
    The scheduled time is interpreted by the Scheduler class, generally seconds.
    The task is any function that will be given the scheduler's notion of elapsed time.
    If provided, the delay is a reschedule delay. The limit is the number of
    times the task can be run, default is once only.
    """

    scheduled: int
    callback: Callback = field(compare=False)
    delay: int = field(default=0, compare=False)
    limit: int = field(default=1, compare=False)
    def repeat(self, current_time: int) -> Optional["Task"]:
        if self.delay > 0 and self.limit > 2:
            return Task(
                current_time + self.delay,
                cast(Callback, self.callback), # type: ignore [misc]
                self.delay,
                self.limit - 1,
            )
        elif self.delay > 0 and self.limit == 2:
            return Task(
                current_time + self.delay,
                cast(Callback, self.callback), # type: ignore [misc]
            )
        else:
            return None