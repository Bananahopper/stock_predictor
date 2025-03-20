import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go


class Visualizer:
    def __init__(self, ticker_name, data, save_path="./visualizations"):
        self.ticker_name = ticker_name
        self.data = data
        self.save_path = save_path

        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def plot_boxplots(self, dataframe):
        for column in dataframe.columns:
            fig = dataframe.boxplot(column=column)
            fig.get_figure().savefig(f"{self.save_path}/{column}_boxplot.png")

    def plot_candlestick(self, data):

        layout = go.Layout(
            title=f"{self.ticker_name} candlestick chart",
            xaxis=dict(title="Date"),
            yaxis=dict(title="Price"),
        )

        candlesticks = go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Candlestick",
            increasing=dict(line=dict(color="green")),
            decreasing=dict(line=dict(color="red")),
        )

        fig = go.Figure(data=[candlesticks], layout=layout)
        fig.update_layout(xaxis_rangeslider_visible=False)
        fig.write_image(f"{self.save_path}/candlestick.pdf")
