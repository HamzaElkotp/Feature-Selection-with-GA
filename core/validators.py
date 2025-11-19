def validate_config(config):
    if config.elitism_percent + config.mutation_percent > 25:
        raise ValueError("Elitism + Mutation cannot exceed 25%")