from pathlib import Path

def validate_config(config):
    if config.elitism_percent + config.mutation_percent > 25:
        raise ValueError("Elitism + Mutation cannot exceed 25%")
    
def validate_dataset_path_inline(path_str: str):
    """
    Validate dataset path and return a tuple (is_valid, message).

    Returns:
        (bool, str): True if valid, otherwise False and an error text.
    """
    if not path_str or not str(path_str).strip():
        return False, "Please provide a dataset path."

    p = Path(path_str)

    if not p.exists():
        return False, "File does not exist."

    if not p.is_file():
        return False, "Selected path is not a file."

    if any(part.startswith('.') for part in p.parts):
        return False, "Hidden files or folders are not allowed."

    return True, ""