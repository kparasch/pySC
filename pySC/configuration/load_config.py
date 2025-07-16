import yaml

def load_yaml(file_path: str) -> dict:
    """
    Load a YAML file and return its contents as a dictionary.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Contents of the YAML file.
    """
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
