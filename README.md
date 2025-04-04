# Stock Predictor

This repository provides code for a simple Stock Prediction model. The model uses data from Yahoo Finance to make predictions about the closing price of a ticker. This repository is intended to be an educational tool to deepen one's knowledge on how an LSTM model is trained and on how it can be applied to make predictions on time series data.

## DISCLAIMER - Is this model financial advice or to be trusted to make predictions in the real stock market?

Absolutely not.

## Usage

Firstly, create an environment from the environment file.

<conda env create --file=environment.yaml>

Then, edit the configs/config.json file to your preference. If you want to train a new model on a new ticker, edit the ticker key and set the mode to "TRAIN". Else you can use the provided model that was trained on ING stock data.

Finally, run the main.py file.

<python src/main.py --config='path/to/config/file'>

## Exploratory data analysis

From Yahoo Finance you can extract 7 variables about any ticker. Closeing price, Opening price, Lowest daily price, Highest daily price, Daily Volume, Dividends, and Splits.

Since Dividends and Splits rarely have any information we can remove these columns from the dataset.

The Open, Close, Low, and High prices follow very similar distributions as can be seen in these boxplots (only close boxplot shown to save space. The others are under visuals):

![Close_box](visuals/Close_boxplot.png?width=300)


These distributions are all right skewed, meaning that there are several outliers in each column. This is to be expected with financial data as the market can fluctuate quite severly in a day. These distributions being so remarkably similar indicates that the ING stock has stable trading patterns. This should make it an easy stock to create predictions for this stock.

The volume distribution is also right skewed. However, the scale is distinctly different from that of the other variables, meaning that this will have to be accounted for during data transformation or it could impact the performance of the model.

The similarities and distinctions described above can also be seen when you plot the data over time.

![Over_time](visuals/over_time.png)

## The model

The model is a slightly modified LSTM architecture. The basic LSTM is provided by PyTorch (torch.nn.LSTM). A fully connected layer was appended to the end of the LSTM, as a minor modification.

## Accuracy Measures

To measure the accuracy of the model we used 2 measurements: Mean Absolute Percentage Error; Directional Accuracy

### Mean Absolute Percentage Error

To intuitively understand by how much the model prediction deviates from the expected value, we used the Mean Absolute Percentage Error. Given by the following equation:

$$MAPE = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{x_i - y_i}{x_i} \right| \times 100$$

,where

$n$ = the number of prediction, label tuples in the batch

$x_i$ = the label value

$y_i$ = the prediction value

### Directional Accuracy

The Directional Accuracy shows how often the model correctly predicts if the stock price is increasing or decreasing, expressed in percentages. It is given by the following equation:

$$Directional Accuracy = \frac{1}{n-1} \sum_{i=1}^{n-1} \mathbb{1}[\text{sign}(y_{i+1} - y_i) = \text{sign}(x_{i+1} - x_i)]$$

,where

$n$ = number of tuples in the batch

$y_i$ = the prediction value

$x_i$ = the label value

## Results

The final model trained on ING data achieves a DA of 89% and a MAPE of 1.14 on the test set.

![Predictions](visuals/predictions_vs_labels_epoch_1.png)

## Acknowledgments

Melissa Monfared's kaggle notebook : https://www.kaggle.com/code/melissamonfared/google-s-stock-price-prediction-lstm/notebook
