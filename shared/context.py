# shared/context.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any

from interfaces.enums import RunMode
from interfaces.types import GAParameters


@dataclass
class AppContext:
    """Holds values shared across pages and controllers.

    This version stores typed values from `interfaces` (enums and dataclasses)
    instead of plain strings so the rest of the app can use typed objects.
    """
    mode: Optional[RunMode] = None
    dataset_path: Optional[Path] = None
    ga_parameters: Optional[GAParameters] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)

    # for validation or debugging
    def summary(self):
        return {
            "mode": self.mode.name if self.mode else None,
            "dataset_path": str(self.dataset_path) if self.dataset_path else None,
            "ga_parameters": self.ga_parameters.__dict__ if self.ga_parameters else None,
            "extra_data": self.extra_data,
        }