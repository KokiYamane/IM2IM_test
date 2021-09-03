from torch import nn

class LSTMBlock(nn.Module):
    def __init__(self, input_dim, output_dim, LSTM_dim=100, LSTM_layer_num=2, dropout=0):
        super().__init__()

        self.LSTM_dim = LSTM_dim
        self.LSTM_layer_num = LSTM_layer_num

        self.lstm = nn.LSTM(input_dim, LSTM_dim,
                            num_layers=LSTM_layer_num,
                            batch_first=True, dropout=dropout)
        self.linear = nn.Linear(LSTM_dim, output_dim)

    def forward(self, state, memory=None):
        y, (h, c) = self.lstm(state, memory)
        state_hat = self.linear(y)

        if memory == None:
            return state_hat
        else:
            return state_hat, (h, c)
            