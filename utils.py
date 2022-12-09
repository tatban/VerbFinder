from typing import Dict
import warnings
import yaml


def read_yaml(file_path: str) -> Dict:
    """
    Reads configuration from config.yaml file in a dictionary.
    :param file_path: string path of the config.yaml file
    :return: a nested dictionary of the config keys and corresponding values.
    """
    assert file_path.split(".")[-1].lower() == "yaml", "config file needs to be  a .yaml file"
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        warnings.warn(
            f"Could not load config file due to following exception. Defaults values "
            f"will be used:\n{e}")
        return {}  # returning empty dictionary for default values of the configs
