import torch


def get_loss(loss_name):
    if loss_name == "MSE":
        return torch.nn.MSELoss()
    if loss_name == "MAE":
        return torch.nn.L1Loss()
