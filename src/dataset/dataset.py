import yfinance as yf

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

    def _process_data(self, data):
        
        if data.isnull().values.any():
            print('Data contains missing values. Replacing NaN with 0.')
            data.fillna(0, inplace=True)
        return data

    def get_dataset(self):
        tckr = yf.Ticker(self.ticker)
        tckr_history = tckr.history(start=self.start_date, end=None)
        data = self._process_data(tckr_history)
        return data
    

if __name__ == '__main__':
    data = TickerDataset('ING', '2010-01-01').get_dataset()
    print(data)

