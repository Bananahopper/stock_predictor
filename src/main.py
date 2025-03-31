import torch
from dataset.base_dataset import BaseTickerDataset
from dataset.lstm_dataset import LSTMTickerDataset
from loss import get_loss
from model import get_model
from optimizer import get_optimizer
from test import Tester
from visualization.visualizations import Visualizer
from utils.utils import read_config_file, train_test_val_split
import argparse
from train import Trainer
from utils.utils import initialize_wandb


def main():
    parser = argparse.ArgumentParser(description="Visualize stock data")
    parser.add_argument("--config", type=str, help="Path to the config file")
    args = parser.parse_args()

    config = read_config_file(args.config)

    initialize_wandb(
        project_name=config["WANDB_LOG"]["PROJECT_NAME"],
        run_name=config["WANDB_LOG"]["RUN_NAME"],
        notes=config["WANDB_LOG"]["NOTES"],
        tags=config["WANDB_LOG"]["TAGS"],
    )

    data, data_moving_average = BaseTickerDataset(
        config["TICKER_SELECTION"]["TICKER"],
        config["TICKER_SELECTION"]["START_DATE"],
    ).get_dataset()

    model = get_model(
        config["MODEL"]["INPUT_SIZE"],
        config["MODEL"]["HIDDEN_SIZE"],
        config["MODEL"]["OUTPUT_SIZE"],
    )

    optimizer = get_optimizer(model.parameters(), config["OPTIMIZER"]["LEARNING_RATE"])
    loss = get_loss(config["TRAINING"]["LOSS"])

    train_data, val_data, test_data = train_test_val_split(
        data_moving_average,
        config["TRAINING"]["TRAIN_SIZE"],
        config["TRAINING"]["VAL_SIZE"],
    )

    if config["VISUALIZATION"]:

        print("Visualizing ticker data")

        visualizer = Visualizer(config["TICKER_SELECTION"], data)
        visualizer.plot_boxplots()
        visualizer.plot_density()
        visualizer.plot_candlestick()

    if config["MODE"] == "TRAIN":

        print("Initializing training with the following parameters:")
        print(f"Model: {config['MODEL']['NAME']}")
        print(f"Optimizer: {config['OPTIMIZER']['NAME']}")
        print(f"Loss: {loss}")
        print(f"Train size: {len(train_data)}")
        print(f"Validation size: {len(val_data)}")
        print(f"Batch size: {config['TRAINING']['BATCH_SIZE']}")
        print(f"Epochs: {config['TRAINING']['EPOCHS']}")
        print(f"Learning rate: {config['OPTIMIZER']['LEARNING_RATE']}")

        train_dataset = LSTMTickerDataset(train_data)
        val_dataset = LSTMTickerDataset(val_data)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, config["TRAINING"]["BATCH_SIZE"], shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, config["TRAINING"]["BATCH_SIZE"]
        )

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss=loss,
            train_loader=train_loader,
            validation_loader=val_loader,
            num_epochs=config["TRAINING"]["EPOCHS"],
            device="cuda" if torch.cuda.is_available() else "cpu",
            result_path=config["TRAINING"]["RESULT_PATH"],
            checkpoint=config["TRAINING"]["CHECKPOINT"],
        )

        trainer.fit()

    if config["MODE"] == "TEST":

        print("Initializing testing with the following parameters:")
        print(f"Test size: {len(test_data)}")
        print(f"Batch size: {config['TRAINING']['BATCH_SIZE']}")

        test_dataset = LSTMTickerDataset(test_data)

        test_loader = torch.utils.data.DataLoader(
            test_dataset, config["TRAINING"]["BATCH_SIZE"]
        )

        tester = Tester(
            model=model,
            train_loader=test_loader,
            device="cuda" if torch.cuda.is_available() else "cpu",
            result_path=config["TRAINING"]["RESULT_PATH"],
            checkpoint=config["TRAINING"]["CHECKPOINT"],
        )

        tester.test()

        print("Testing completed.")


if __name__ == "__main__":
    main()
