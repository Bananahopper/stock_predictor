import os
import matplotlib.pyplot as plt

class Visualizer():
    def __init__(self, data, save_path="./visualizations"):
        self.data = data
        self.save_path = save_path

        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def plot_boxplots(self, dataframe):
        for column in dataframe.columns:
            fig = dataframe.boxplot(column=column)
            fig.get_figure().savefig(f"{self.save_path}/{column}_boxplot.png")

    