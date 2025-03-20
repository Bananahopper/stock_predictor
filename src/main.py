from dataset.dataset import TickerDataset
from visualization.visualizations import Visualizer
from utils.utils import read_config_file
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Visualize stock data")
    parser.add_argument("--config", type=str, help="Path to the config file")
    args = parser.parse_args()

    config = read_config_file(args.config)

    data = TickerDataset(config["TICKER"], config["START_DATE"]).get_dataset()
    visualizer = Visualizer(config["TICKER"], data)
    visualizer.plot_boxplots(data)
    visualizer.plot_candlestick(data)
