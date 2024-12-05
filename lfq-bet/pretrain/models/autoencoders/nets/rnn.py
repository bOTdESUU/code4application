import torch

import torch.nn as nn

class NavieRNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        hidden = self.init_hidden()

        output, hidden = self.rnn(input, hidden)
        output = self.fc(output[-1])

        return output

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)