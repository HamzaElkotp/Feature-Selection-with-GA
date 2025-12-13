# Inputs
- Dataset file path as csv: str
- Mode: Enum[`DIM (Deterministic Input Mode)`, `EBM (Empirical Benchmarking Mode)`]
- Selection method: Enum[`Roulette Wheel Selection`, `Ranking`, `Tournament Selection`]
- Crossover method: Enum[`Sigle-Point Corssover`, `Multi-Point Crossover`]
- Mutation method: Enum[`Bit Flip`, `Inversiton`]
- Elitism Percent: (0.00, 25)
- Mutation Percent Percent: (0.00, 25)
- Alpha: float(0.00-2.00)
- Beta: float(0.00-2.00)
- Termination condition: {
    condition: Enum[`After N Generations`, `After N Seconds`, `After Fitness Reaches N`, `No Improvment Since N Generations`],
    N: float
}
> **Note:** If the mode is DIM the rest of the properties will be return normally otherwise if the mode
is EBM the properties will have default values except for the alpha and beta the values will be set to 0
and the core logic should set them to different values for each try
# Outputs

> **Note:** In all the following outputs, it's considered that after running the run_ga function it will return a RunResult, which has "best_genome" attribute and generations dict with the following structure
```python
generations = {
    0: [
        {"parent_id": 0, "id": 1, "fitness": 2.0, "accuracy": 0.34, "features": ["Age", "Blood", "Fever"]},
        {"parent_id": 0, "id": 2, "fitness": 2.0, "accuracy": 0.34, "features": ["Age", "Blood", "Fever"]},
        {"parent_id": 0, "id": 3, "fitness": 2.0, "accuracy": 0.34, "features": ["Age", "Blood", "Fever"]}
    ],
    1: [
        {"parent_id": 1, "id": 4, "fitness": 2.0, "accuracy": 0.34, "features": ["Age", "Blood", "Fever"]},
        {"parent_id": 1, "id": 5, "fitness": 2.0, "accuracy": 0.34, "features": ["Age", "Blood", "Fever"]},
        {"parent_id": 2, "id": 6, "fitness": 2.0, "accuracy": 0.34, "features": ["Age", "Blood", "Fever"]},
        {"parent_id": 2, "id": 7, "fitness": 2.0, "accuracy": 0.34, "features": ["Age", "Blood", "Fever"]},
    ],
}
```
### 1. Suggest the best gene of features 

### 2. Compare average fitness across generations
![1](https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs41598-024-73335-6/MediaObjects/41598_2024_73335_Fig11_HTML.png)
