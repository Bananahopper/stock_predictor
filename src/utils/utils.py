from datetime import datetime, UTC
import json
import os
import torch
import wandb


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


def train_test_val_split(data, train_split_size, val_split_size):

    train_data = data[: int(len(data) * train_split_size)]
    val_data = data[
        int(len(data) * train_split_size) : int(
            len(data) * (train_split_size + val_split_size)
        )
    ]
    test_data = data[int(len(data) * (train_split_size + val_split_size)) :]

    return train_data, val_data, test_data


def initialize_wandb(project_name, run_name, notes="", tags=""):
    wandb.init(
        project=project_name,
        notes=notes,
        name=f'{run_name}_{datetime.now(UTC).strftime("%Y-%m-%d")}',
        tags=tags,
    )
