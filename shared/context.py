# shared/context.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class AppContext:
    """Holds values shared across pages and controllers."""
    mode: Optional[str] = None                 # deterministic / non-deterministic
    dataset_path: Optional[Path] = None        # path to CSV or dataset file
    extra_data: Dict[str, Any] = field(default_factory=dict)

    # for validation or debugging
    def summary(self):
        return {
            "mode": self.mode,
            "dataset_path": str(self.dataset_path) if self.dataset_path else None,
            "extra_data": self.extra_data,
        }