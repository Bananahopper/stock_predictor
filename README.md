# Stock Predictor

This repository provides code for a simple Stock Prediction model. The model uses data from Yahoo Finance to make predictions about the closing price of a ticker.

## Usage

## Accuracy Measures

To measure the accuracy of the model we used 2 measurements: Mean Absolute Percentage Error; Directional Accuracy

### Mean Absolute Percentage Error

To intuitively understand by how much the model prediction deviates from the expected value, we used the Mean Absolute Percentage Error. Given by the following equation:

MAPE = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{x_i - y_i}{x_i} \right| \times 100

,where

n = the number of prediction, label tuples in the batch

x_i = the label value

y_i = the prediction value

### Directional Accuracy 

The Directional Accuracy shows how often the model correctly predicts if the stock price is increasing or decreasing, expressed in percentages. It is given by the following equation:

Directional Accuracy = \frac{1}{n-1} \sum_{i=1}^{n-1} \mathbb{1}[\text{sign}(y_{i+1} - y_i) = \text{sign}(x_{i+1} - x_i)]

,where

n = number of tuples in the batch

y_i = the prediction value

x_i = the label value

## Acknowledgments
Melissa Monfared's kaggle notebook : https://www.kaggle.com/code/melissamonfared/google-s-stock-price-prediction-lstm/notebook
