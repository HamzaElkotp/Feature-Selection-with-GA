from interfaces.types import GAParameters, RunGAParameters, RunGAResult
from abc import ABC, abstractmethod

class GAInterface(ABC):

    @abstractmethod
    def run_ga(self, parms: RunGAParameters) -> RunGAResult:
        """Run the Genetic Algorithm with the given parameters."""
        pass