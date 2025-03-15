from dataset.dataset import TickerDataset
from visualization.visualizations import Visualizer

if __name__ == '__main__':
    data = TickerDataset('ING', '2010-01-01').get_dataset()
    visualizer = Visualizer(data)
    visualizer.plot_boxplots(data)