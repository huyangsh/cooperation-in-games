import torch
import torch.nn as nn

class LSTMQNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, proj_size, output_size, num_layers):
        super(LSTMQNetwork, self).__init__()
        self.lstm_layer = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            proj_size   = proj_size,
            num_layers  = num_layers,
            batch_first = False
        )
        proj_size = hidden_size if proj_size == 0 else proj_size
        self.fc_layer = nn.Linear(proj_size, output_size)
    
    def forward(self, x):
        if len(x.shape) == 2:
            return self.fc_layer(self.lstm_layer(x)[0][-1])
        elif len(x.shape) == 3:
            return self.fc_layer(self.lstm_layer(x)[0][:,-1])