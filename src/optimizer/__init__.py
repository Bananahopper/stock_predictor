import torch


def get_optimizer(model_params, lr):
    return torch.optim.Adam(model_params, lr=lr)
