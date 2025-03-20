from src.dataset.base_dataset import BaseTickerDataset
import numpy as np


class LSTMTickerDataset(BaseTickerDataset):
    """
    This class gets ticker
    data from yahoo finance and converts it to a usable dataset for machine learning."
    """

    def __init__(self, ticker: str, start_date: str, moving_average: int = 10):

        super().__init__(
            ticker=ticker, start_date=start_date, moving_average=moving_average
        )

    def __len__(self):
        return len(self.data_moving_average)

    def __getitem__(self, idx):

        all_values = self.data_moving_average.iloc[idx].values
        close_idx = self.data_moving_average.columns.get_loc("Close")
        input_data = np.delete(all_values, close_idx)
        label = all_values[close_idx]

        return input_data, label


if __name__ == "__main__":
    lstm_ticker = LSTMTickerDataset("ING", "2010-01-01")
    print(lstm_ticker[13])
