import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from utils.utils import create_folder_structure


class Visualizer:
    def __init__(self, ticker_name, data, save_path="./visualizations"):
        self.ticker_name = ticker_name
        self.data = data
        self.save_path = save_path

        # Create folders for all the visualizations
        self.folders = ["boxplots", "densities", "candlesticks"]
        create_folder_structure(self.save_path, self.folders)

    def plot_boxplots(self):
        # Filter only numeric columns and exclude specified columns
        numeric_columns = [
            col
            for col in self.data.columns
            if np.issubdtype(self.data[col].dtype, np.number)
        ]

        for column in numeric_columns:
            # Create figure and axis with a reasonable size
            plt.figure(figsize=(10, 6))

            # Calculate statistics
            stats = {
                "Mean": self.data[column].mean(),
                "Std Dev": self.data[column].std(),
                "Min": self.data[column].min(),
                "Max": self.data[column].max(),
                "Median": self.data[column].median(),
            }

            # Use matplotlib's boxplot with default settings for outliers
            # This only plots the outliers and not the points within quartiles
            boxplot = plt.boxplot(
                self.data[column], vert=True, patch_artist=True, widths=0.5
            )

            # Customize boxplot colors
            for box in boxplot["boxes"]:
                box.set(facecolor="#9ecae1", alpha=0.8)
            for whisker in boxplot["whiskers"]:
                whisker.set(color="#3182bd", linewidth=1.5)
            for cap in boxplot["caps"]:
                cap.set(color="#3182bd", linewidth=1.5)
            for median in boxplot["medians"]:
                median.set(color="#08519c", linewidth=2)
            for flier in boxplot["fliers"]:
                flier.set(
                    marker="o",
                    markerfacecolor="#e34a33",
                    markeredgecolor="none",
                    markersize=5,
                    alpha=0.7,
                )

            # Create stats text
            stats_text = "\n".join(
                [f"{key}: {value:.4f}" for key, value in stats.items()]
            )

            # Add text box with statistics
            plt.text(
                0.05,
                0.95,
                stats_text,
                transform=plt.gca().transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            # Set title and labels
            plt.title(f"Boxplot of {column} - {self.ticker_name}", fontsize=14, pad=20)
            plt.ylabel(column, fontsize=12)
            plt.xlabel("")
            plt.xticks([])
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.axhline(
                y=stats["Mean"], color="red", linestyle="--", linewidth=1.5, alpha=0.8
            )

            # Annotate mean line
            plt.text(
                1.05,
                stats["Mean"],
                "Mean",
                verticalalignment="bottom",
                color="red",
                fontweight="bold",
            )
            plt.tight_layout()

            # Save figure with high DPI for better quality
            plt.savefig(
                f"{self.save_path}/{self.folders[0]}/{column}_boxplot.png",
                dpi=300,
                bbox_inches="tight",
            )

    def plot_density(self):

        # Filter only numeric columns
        numeric_columns = [
            col
            for col in self.data.columns
            if np.issubdtype(self.data[col].dtype, np.number)
        ]

        for column in numeric_columns:
            # Create figure and axis with a reasonable size
            plt.figure(figsize=(10, 6))

            # Calculate statistics
            stats = {
                "Mean": self.data[column].mean(),
                "Std Dev": self.data[column].std(),
                "Min": self.data[column].min(),
                "Max": self.data[column].max(),
                "Median": self.data[column].median(),
            }

            boxplot = plt.boxplot(
                self.data[column], vert=True, patch_artist=True, widths=0.5
            )

            # Customize boxplot colors
            for box in boxplot["boxes"]:
                box.set(facecolor="#9ecae1", alpha=0.8)
            for whisker in boxplot["whiskers"]:
                whisker.set(color="#3182bd", linewidth=1.5)
            for cap in boxplot["caps"]:
                cap.set(color="#3182bd", linewidth=1.5)
            for median in boxplot["medians"]:
                median.set(color="#08519c", linewidth=2)
            for flier in boxplot["fliers"]:
                flier.set(
                    marker="o",
                    markerfacecolor="#bdbdbd",
                    markeredgecolor="none",
                    markersize=4,
                    alpha=0.5,
                )

            # Add individual points (optional and more efficient than swarmplot)
            # Using sampling if the dataset is large to prevent performance issues
            sample_size = min(
                100, len(self.data)
            )  # Limit to at most 100 points for display
            if len(self.data) > sample_size:
                sampled_data = self.data[column].sample(sample_size)
            else:
                sampled_data = self.data[column]

            # Add sampled points as a scatter plot
            y_positions = sampled_data.values
            x_positions = np.random.normal(1, 0.04, size=len(y_positions))
            plt.scatter(x_positions, y_positions, s=10, color="#636363", alpha=0.4)

            # Create stats text
            stats_text = "\n".join(
                [f"{key}: {value:.4f}" for key, value in stats.items()]
            )

            plt.text(
                0.05,
                0.95,
                stats_text,
                transform=plt.gca().transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            # Set title and labels
            plt.title(f"Boxplot of {column} - {self.ticker_name}", fontsize=14, pad=20)
            plt.ylabel(column, fontsize=12)
            plt.xlabel("")
            plt.xticks([])
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.axhline(
                y=stats["Mean"], color="red", linestyle="--", linewidth=1.5, alpha=0.8
            )

            # Annotate mean line
            plt.text(
                1.05,
                stats["Mean"],
                "Mean",
                verticalalignment="bottom",
                color="red",
                fontweight="bold",
            )

            # Tight layout
            plt.tight_layout()

            # Save figure with high DPI for better quality
            plt.savefig(
                f"{self.save_path}/{self.folders[0]}/{column}_boxplot.png",
                dpi=300,
                bbox_inches="tight",
            )

    def plot_candlestick(self):

        layout = go.Layout(
            title=f"{self.ticker_name} candlestick chart",
            xaxis=dict(title="Date"),
            yaxis=dict(title="Price"),
        )

        candlesticks = go.Candlestick(
            x=self.data.index,
            open=self.data["Open"],
            high=self.data["High"],
            low=self.data["Low"],
            close=self.data["Close"],
            name="Candlestick",
            increasing=dict(line=dict(color="green")),
            decreasing=dict(line=dict(color="red")),
        )

        fig = go.Figure(data=[candlesticks], layout=layout)
        fig.update_layout(xaxis_rangeslider_visible=False)
        fig.write_image(f"{self.save_path}/{self.folders[2]}/candlestick.pdf")
