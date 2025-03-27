import numpy as np
import torch
import matplotlib.pyplot as plt
import os


def plot_predictions_and_labels(dataloader, model, epoch, result_path, device):
    """
    Plot the predictions and labels from a PyTorch DataLoader

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader with input data
        model (torch.nn.Module): PyTorch model
        epoch (int): Current epoch number
        result_path (str): Path to save the plot
        device (torch.device): Device to run the model on
    """

    os.makedirs(result_path, exist_ok=True)

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            predictions = model(inputs.to(device)).squeeze()

            # Keep batches together for proper sequence
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(targets.cpu().numpy())

    # Concatenate batches in the correct order
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)

    # Calculate metrics for the plot title
    mape = np.mean(np.abs((all_labels - all_predictions) / all_labels)) * 100

    plt.figure(figsize=(12, 6))
    plt.plot(all_predictions, label="True Data", alpha=0.7)
    plt.plot(all_labels, label="Model Predictions", alpha=0.7)
    plt.legend()
    plt.title(f"Predictions vs True Data (Epoch {epoch}, MAPE: {mape:.2f}%)")
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{result_path}/predictions_vs_labels_epoch_{epoch}.png")
