from dataset.base_dataset import BaseTickerDataset
from dataset.lstm_dataset import LSTMTickerDataset
from visualization.visualizations import Visualizer
from utils.utils import read_config_file, train_test_val_split, get_dataloader
import argparse


def main():
    parser = argparse.ArgumentParser(description="Visualize stock data")
    parser.add_argument("--config", type=str, help="Path to the config file")
    args = parser.parse_args()

    config = read_config_file(args.config)

    data, data_moving_average = BaseTickerDataset(
        config["TICKER_SELECTION"]["TICKER"],
        config["TICKER_SELECTION"]["START_DATE"],
    ).get_dataset()

    if config["VISUALIZATION"]:

        visualizer = Visualizer(config["TICKER_SELECTION"], data)
        visualizer.plot_boxplots()
        visualizer.plot_density()
        visualizer.plot_candlestick()

    train_data, val_data, test_data = train_test_val_split(
        data_moving_average,
        config["TRAINING"]["TRAIN_SIZE"],
        config["TRAINING"]["VAL_SIZE"],
    )

    if config["MODE"] == "TRAIN":

        train_dataset = LSTMTickerDataset(train_data)
        val_dataset = LSTMTickerDataset(val_data)

        train_loader = get_dataloader(train_dataset, config["TRAINING"]["BATCH_SIZE"])
        val_loader = get_dataloader(val_dataset, config["TRAINING"]["BATCH_SIZE"])

        print(len(train_loader))
        for item in train_loader:
            print(item[0].shape)

    if config["MODE"] == "TEST":

        test_dataset = LSTMTickerDataset(test_data)

        test_loader = get_dataloader(test_data, config["TRAINING"]["BATCH_SIZE"])


if __name__ == "__main__":

    main()
