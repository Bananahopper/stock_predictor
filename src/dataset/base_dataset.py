import yfinance as yf
import os
import io
from contextlib import redirect_stdout


class BaseTickerDataset:
    """
    This class gets ticker data from yahoo finance and converts it to a usable dataset for machine learning.

    Args:
    ticker (str): The ticker symbol of the stock.
    start_date (str): The start date of the data to get in the format 'YYYY-MM-DD'.
    """

    def __init__(self, ticker: str, start_date: str, moving_average: int = 10):
        self.ticker = ticker
        self.start_date = start_date
        self.moving_average = moving_average

        self.data, self.data_moving_average = self.get_dataset()

    def _check_dataset(self, data):

        log_path = "./logs"
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        shape = data.shape
        nan_count = data.isnull().sum()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            data.info()
        info_output = buffer.getvalue()

        with open(f"{log_path}/{self.ticker}_log.txt", "w") as f:
            f.write(f"Shape of dataset: \n {shape}\n\n")
            f.write(f"Number of NaN values: \n {nan_count}\n\n")
            f.write(
                f"Info: \n {info_output}\
            "
            )

    def _process_and_log_data(self, data):
        self._check_dataset(data)

        # Placeholder for when we might need to normalize, clean or preprocess the data

        # If a column has NAN values, fill them with 0
        data = data.fillna(0)

        # Remove stock splits and dividens column if more than 99% of the values are 0
        if (data["Stock Splits"] == 0).sum() / len(data) > 0.99:
            data = data.drop(columns=["Stock Splits"])
        if (data["Dividends"] == 0).sum() / len(data) > 0.95:
            data = data.drop(columns=["Dividends"])

        # Set date to YYYY-MM-DD format
        data = data.reset_index()
        data["Date"] = data["Date"].dt.strftime("%Y-%m-%d")
        data.set_index("Date", inplace=True)

        return data

    def _get_moving_average_dataset(self, data):
        data_moving_average = data.copy(deep=True)

        data_moving_average["Moving Average"] = (
            data_moving_average["Close"].rolling(window=self.moving_average).mean()
        )
        data_moving_average["Moving Average"] = data_moving_average[
            "Moving Average"
        ].fillna(data_moving_average["Close"])

        data_moving_average.reset_index(inplace=True)
        data_moving_average["Volume"] = data_moving_average["Volume"].astype(float)

        return data_moving_average

    def get_dataset(self):
        tckr = yf.Ticker(self.ticker)
        tckr_history = tckr.history(start=self.start_date, end=None)
        data = self._process_and_log_data(tckr_history)
        data_moving_average = self._get_moving_average_dataset(data)
        return data, data_moving_average


if __name__ == "__main__":
    data, data_moving_average = BaseTickerDataset("ING", "2010-01-01").get_dataset()
    print(data_moving_average.head(15))
