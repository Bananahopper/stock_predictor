import json
import os


def read_config_file(config_file):
    """
    Reads a json config file and returns a dictionary

    Args:
        config_file (str): Path to the config file
    """
    with open(config_file, "r") as f:
        config = json.load(f)
    return config


def create_folder_structure(save_path: str, list_of_subfolders=[]):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if list_of_subfolders != []:
        for folder in list_of_subfolders:
            if not os.path.exists(save_path + "/" + folder):
                os.makedirs(save_path + "/" + folder)
