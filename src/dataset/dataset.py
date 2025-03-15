import yfinance as yf
import os
import io
from contextlib import redirect_stdout

class TickerDataset():
    """
    This class gets ticker data from yahoo finance and converts it to a usable dataset for machine learning.

    Args:
    ticker (str): The ticker symbol of the stock.
    start_date (str): The start date of the data to get in the format 'YYYY-MM-DD'.
    """
    def __init__(self, ticker: str, start_date: str):
        self.ticker = ticker
        self.start_date = start_date

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
            f.write(f"Info: \n {info_output}\
            ")

    def _process_and_log_data(self, data):
        self._check_dataset(data)

        # Placeholder for when we might need to normalize, clean or preprocess the data

        return data

    def get_dataset(self):
        tckr = yf.Ticker(self.ticker)
        tckr_history = tckr.history(start=self.start_date, end=None)
        dataframe = self._process_and_log_data(tckr_history)
        return dataframe
    

if __name__ == '__main__':
    data = TickerDataset('ING', '2010-01-01').get_dataset()
    print(data)

