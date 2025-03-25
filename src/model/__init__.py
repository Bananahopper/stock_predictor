from model.model import LSTM


def get_model(input_size, hidden_size, output_size):
    return LSTM(input_size, hidden_size, output_size)
