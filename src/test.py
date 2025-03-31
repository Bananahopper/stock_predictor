import os

import numpy as np
import torch

from plotting.plot_predictions_and_labels import plot_predictions_and_labels


class Tester:
    def __init__(
        self,
        model,
        train_loader,
        device,
        result_path,
        checkpoint,
    ):
        self.model = model
        self.test_loader = train_loader
        self.device = device
        self.result_path_imgs = os.path.join(result_path, "visualizations/test")
        self.checkpoint = checkpoint

        os.makedirs(self.result_path_imgs, exist_ok=True)

        if self.checkpoint is None:
            raise ValueError("Checkpoint is required for testing.")

    def test(self):
        """
        Test the model on the test dataset.
        """
        self.model.load_state_dict(torch.load(self.checkpoint))
        self.model.to(self.device)
        self.model.eval()

        metrics_sum = {
            "mape": 0.0,
            "directional_accuracy": 0.0,
        }

        for idx, (values, labels) in enumerate(self.test_loader):
            outputs = self.model(values.to(self.device)).squeeze()

            batch_metrics = self.evaluate_accuracy(outputs, labels.to(self.device))
            for key in metrics_sum:
                metrics_sum[key] += batch_metrics[key]

        plot_predictions_and_labels(
            self.test_loader,
            self.model,
            1,
            os.path.join(self.result_path_imgs, "visualizations/validation"),
            self.device,
        )

        for key in metrics_sum:
            metrics_sum[key] /= len(self.test_loader)

        print(f"Test MAPE: {metrics_sum['mape']:.2f}%")
        print(f"Test Directional Accuracy: {metrics_sum['directional_accuracy']:.2f}")

    def evaluate_accuracy(self, outputs, labels):

        with torch.no_grad():

            mape = torch.mean(torch.abs((labels - outputs) / labels)) * 100

            if len(outputs) > 1:
                pred_direction = torch.diff(outputs) > 0
                true_direction = torch.diff(labels) > 0
                direction_match = torch.sum(pred_direction == true_direction).item()
                directional_accuracy = direction_match / (len(labels) - 1)
            else:
                directional_accuracy = 0.0

        return {
            "mape": mape.item(),
            "directional_accuracy": directional_accuracy,
        }
