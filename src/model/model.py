import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        x = self.fc(hidden[-1])
        return x


if __name__ == "__main__":
    model = LSTM(6, 64, 1)
    x = torch.randn(32, 10, 6)
    output = model(x)
    print(output.shape)
