import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import MinMaxScaler

from dataset.base_dataset import BaseTickerDataset


class LSTMTickerDataset(Dataset):
    """
    Dataset for LSTM-based stock price prediction.

    This class takes the processed ticker data and creates sequences
    for training an LSTM model to predict the Close price.

    Args:
        data (pd.DataFrame): DataFrame with columns including Date, Open, High, Low, Close, Volume, Moving Average
        sequence_length (int): Number of time steps in each input sequence
        target_column (str): Column to predict (default: "Close")
    """

    def __init__(
        self,
        data: pd.DataFrame,
        sequence_length: int = 10,
        target_column: str = "Close",
    ):

        self.data = self._normalize_data(data.copy().reset_index(drop=True))
        self.sequence_length = sequence_length
        self.target_column = target_column
        self.feature_columns = [col for col in self.data.columns if col != "Date"]
        self.target_idx = self.feature_columns.index(target_column)

    def __len__(self):
        return max(0, len(self.data) - self.sequence_length)

    def __getitem__(self, idx):

        sequence = self.data.iloc[idx : idx + self.sequence_length]
        features = sequence[self.feature_columns].values
        target = features[-1, self.target_idx]
        inputs = np.delete(features, self.target_idx, axis=1)

        input_tensor = torch.tensor(inputs, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)

        return input_tensor, target_tensor

    def _normalize_data(self, data):
        """
        Normalize the data using min-max scaling.
        """
        scaler = MinMaxScaler(feature_range=(0, 2)).fit(data.Low.values.reshape(-1, 1))
        data["Open"] = scaler.transform(data.Open.values.reshape(-1, 1))
        data["High"] = scaler.transform(data.High.values.reshape(-1, 1))
        data["Low"] = scaler.transform(data.Low.values.reshape(-1, 1))
        data["Close"] = scaler.transform(data.Close.values.reshape(-1, 1))
        # Additional log transformation for the Volume column
        data["Volume"] = np.log10(data.Volume)
        data["Volume"] = scaler.transform(data.Volume.values.reshape(-1, 1))
        data["Moving Average"] = scaler.transform(
            data["Moving Average"].values.reshape(-1, 1)
        )
        return data


if __name__ == "__main__":
    base_dataset = BaseTickerDataset(ticker="AAPL", start_date="2020-01-01")
    df = base_dataset.data_moving_average

    lstm_dataset = LSTMTickerDataset(data=df)
    print(lstm_dataset[0][0])
