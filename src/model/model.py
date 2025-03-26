import torch
import torch.nn as nn

from dataset.base_dataset import BaseTickerDataset
from dataset.lstm_dataset import LSTMTickerDataset
from utils.utils import train_test_val_split


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        x = self.fc(output[:, -1, :])
        return x


if __name__ == "__main__":

    # Add this right after creating your model
    def check_weights(model):
        for name, param in model.named_parameters():
            print(f"{name} - mean: {param.mean().item()}, std: {param.std().item()}")
            # Check if weights are all the same
            if param.dim() > 1:
                print(f"  All same? {torch.all(param == param[0, 0])}")

    model = LSTM(5, 32, 1)
    check_weights(model)

    data, data_moving_average = BaseTickerDataset(
        "ING",
        "2019-01-01",
    ).get_dataset()

    train_data, val_data, test_data = train_test_val_split(
        data_moving_average,
        0.7,
        0.2,
    )

    print(train_data.head())

    train_dataset = LSTMTickerDataset(train_data)

    train_loader = torch.utils.data.DataLoader(train_dataset, 16)

    for x, y in train_loader:
        # Print the shape of the input tensor
        print("Input shape:", x.shape)

        # Check if all inputs are identical
        all_same = torch.all(x[0] == x[1:])
        print("All inputs identical:", all_same)

        # Check variance across batch dimension
        variance = torch.var(x, dim=0).mean()
        print("Average variance across batch:", variance.item())

        # Print a few samples
        print("First sample:\n", x[0])
        print("Second sample:\n", x[1])

        # Then check the output
        output = model(x)
        print("Output:", output.squeeze())
        break
