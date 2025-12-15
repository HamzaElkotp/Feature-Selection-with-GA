from core.GA_functions import Merged_GA
from interfaces.types import GAParameters, RunGAParameters, RunGAResult
from abc import ABC, abstractmethod
from typing import Callable, Optional


class GAInterface(ABC):

    @abstractmethod
    def run_ga(self, parms: RunGAParameters, on_complete: Optional[Callable[[Merged_GA, Merged_GA], None]] = None) -> None:
        """Run the Genetic Algorithm with the given parameters.

        The implementation should call `on_complete(result)` when the GA finishes
        if an `on_complete` callback is provided. Implementations are free to
        execute synchronously (caller may call them from a background thread)
        or manage their own threading. The method returns None and signals
        completion via the callback.
        """
        pass