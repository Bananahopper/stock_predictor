import json


def read_config_file(config_file):
    """
    Reads a json config file and returns a dictionary

    Args:
        config_file (str): Path to the config file
    """
    with open(config_file, "r") as f:
        config = json.load(f)
    return config
