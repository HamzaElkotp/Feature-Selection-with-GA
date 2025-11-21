from pydantic import BaseModel, Field
from pathlib import Path

class InputConfig(BaseModel):
    mode: str = "deterministic"
    dataset_path: Path | None = None
    termination_condition: str = "max_generation"
    select_method: str = "Roulette"
    cross_method: str = "Uniform"
    mutation_method: str = "swap"
    elitism_percent: float = Field(5,ge=0, le=25)
    mutation_percent: float = Field(5,ge=0, le=25)
    alpha: float = Field(1.0,ge=0, le=2)
    beta: float = Field(1.0,ge=0, le=2)