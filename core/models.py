from pydantic import BaseModel, Field
from pathlib import Path

class DeterministicInputConfig(BaseModel):
    mode: str
    dataset_path: Path
    select_method: str
    cross_method: str
    mutation_method: str
    elitism_percent: float = Field(ge=0, le=25)
    mutation_percent: float = Field(ge=0, le=25)
    alpha: float = Field(ge=0, le=2)
    beta: float = Field(ge=0, le=2)