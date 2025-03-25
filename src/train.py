import copy
import logging
import os
import wandb
import torch
from tqdm import tqdm

from utils.utils import create_folder_structure


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        loss,
        train_loader,
        validation_loader,
        num_epochs,
        device,
        result_path,
        checkpoint,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.num_epochs = num_epochs
        self.device = device
        self.result_path = result_path
        self.result_train_path = os.path.join(result_path, "train")
        self.checkpoint = checkpoint
        self.current_epoch = 0

        create_folder_structure(self.result_path, ["train"])

        self.load_checkpoint()

    def save_model_checkpoint(self):
        """
        Saves the training checkpoint
        """
        # Save checkpoint with model, optimizer, and other info
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        checkpoint_path = os.path.join(
            self.result_train_path, f"model_checkpoint_{self.current_epoch}.pth"
        )

        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self):
        """
        Load the training checkpoint if provided
        """
        if self.checkpoint is not None:
            self.current_epoch = self.checkpoint["epoch"]

    def fit(self):
        self.best_epoch = 0
        self.best_metric = 0.0
        self.best_model = None
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            self.train()
            with torch.no_grad():
                current_val = self.validate()
            if current_val > self.best_metric:
                self.best_metric = current_val
                self.best_epoch = epoch
                self.best_model = copy.deepcopy(self.model)
            self.save_model_checkpoint()
        return self.best_model, self.best_epoch

    def train(self):
        self.model.train()
        self.running_loss = 0.0
        self.avg_epoch_loss = 0.0
        for idx, (values, labels) in tqdm(
            enumerate(self.train_loader), total=len(self.train_loader)
        ):

            self.optimizer.zero_grad()

            outputs = self.model(values.to(self.device)).squeeze()
            loss = self.loss(outputs, labels.to(self.device))

            if torch.isnan(loss):
                raise ValueError("Loss is NaN")

            self.running_loss += loss.item()
            self.avg_epoch_loss += loss.item()
            self.log_train_metrics(idx)
        self.optimizer.zero_grad(set_to_none=True)
        self.avg_epoch_loss /= len(self.train_loader)
        wandb.log(
            {"train_loss": self.avg_epoch_loss, "epoch": self.current_epoch},
            step=self.current_epoch,
        )

    def validate(self):
        self.model.eval()
        self.running_loss = 0.0
        self.avg_epoch_loss = 0.0
        for idx, (values, labels) in tqdm(
            enumerate(self.validation_loader), total=len(self.validation_loader)
        ):
            outputs = self.model(values.to(self.device)).squeeze()
            loss = self.loss(outputs, labels.to(self.device))

            if torch.isnan(loss):
                raise ValueError("Loss is NaN")

            self.running_loss += loss.item()
            self.avg_epoch_loss += loss.item()
        self.avg_epoch_loss /= len(self.validation_loader)
        wandb.log(
            {"val_loss": self.avg_epoch_loss, "epoch": self.current_epoch},
            step=self.current_epoch,
        )
        return self.avg_epoch_loss

    def log_train_metrics(self, idx):
        if idx % self.num_epochs == 0:
            wandb.log(
                {
                    "train_loss": self.running_loss / self.num_epochs,
                    "epoch": self.current_epoch,
                },
                step=self.current_epoch,
            )
            self.running_loss = 0.0
