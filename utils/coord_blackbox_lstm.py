## Modified from https://github.com/xhchrn/MS4L2O 

import math
import torch
import numpy               as np
import torch.nn            as nn
import torch.optim         as optim
import torch.nn.functional as F



class CoordBlackboxLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, layers, **kwargs):
        """
        Coordinate-wise blackbox LSTM that share weights across all coordinates
        of the optimizees.
		An implement of the following paper:
		Andrychowicz et al. (2016) "Learning to learn by gradient descent by gradient descent."
        """
        super().__init__()

        self.input_size  = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        use_bias = True

        self.layers = layers  # Number of layers for LSTM

        self.lstm = nn.LSTM(input_size, hidden_size, layers, bias=use_bias)

        # self.linear = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        # self.linear_out = nn.Linear(hidden_size, output_size, bias=use_bias)
        self.linear = nn.Linear(hidden_size, output_size, bias=use_bias)

        self.state = None

        self.lstm_init_state = 'random'
        self.state_initializer = self.get_state_initializer()

        # self.normalized = False

    @property
    def device(self):
        return self.linear.weight.device

    def get_state_initializer(self):
        if self.lstm_init_state == 'random':
            def initializer(batch_size):
                scale = 1e-0
                hidden = torch.randn(self.layers, batch_size, self.hidden_size)
                cell = torch.randn(self.layers, batch_size, self.hidden_size)
                return ((hidden * scale).to(self.device),
                        (cell * scale).to(self.device))

        elif self.lstm_init_state == 'random-fixed':
            hidden = torch.randn(self.layers, 1, self.hidden_size)
            cell = torch.randn(self.layers, 1, self.hidden_size)
            self.register_buffer('init_hidden', hidden)
            self.register_buffer('init_cell', cell)
            def initializer(batch_size):
                return (self.init_hidden.repeat(1, batch_size, 1),
                        self.init_cell.repeat(1, batch_size, 1))

        elif self.lstm_init_state == 'learned':
            hidden = torch.randn(self.layers, 1, self.hidden_size)
            cell = torch.randn(self.layers, 1, self.hidden_size)
            self.init_hidden = torch.Parameter(hidden)
            self.init_cell = torch.Parameter(cell)
            def initializer(batch_size):
                return (self.init_hidden.repeat(1, batch_size, 1),
                        self.init_cell.repeat(1, batch_size, 1))

        elif self.lstm_init_state == 'zero':
            def initializer(batch_size):
                hidden = torch.zeros(self.layers, batch_size, self.hidden_size)
                cell = torch.zeros(self.layers, batch_size, self.hidden_size)
                return (hidden.to(self.device), cell.to(self.device))

        else:
            raise NotImplementedError

        return initializer

    def reset_state(self, P: int, step_size: float, **kwargs):
        # set initial hidden and cell states
        batch_size = P
        self.step_size = step_size
        self.state = self.state_initializer(batch_size)

    def name(self):
        """
        Function: name
        Purpose : Identify the name of the model
        """
        return 'CoordBlackboxLSTM'


    def forward(
        self, grad,
        reset_state: bool = False,
    ):
        batch_size = grad.shape[0]



        if self.input_size == 1:
            # Here the `grad` is of dimension (batch_size, input_size, 1). Need
            # to reshape it into (1, batch_size, input_size) to be used by LSTM.
            lstm_input = grad.flatten().unsqueeze(0).unsqueeze(-1)
        elif self.input_size == 2:
            raise NotImplementedError
            # flat_grad = grad.flatten().unsqueeze(0).unsqueeze(-1)
            # flat_X = optimizees.X.reshape_as(flat_grad)
            # lstm_input = torch.cat((flat_grad, flat_X), dim=-1)
        else:
            raise NotImplementedError

        # Core update by LSTM.
        output, self.state = self.lstm(lstm_input, self.state)
        update = F.elu(self.linear(output)).reshape_as(grad)
        if not self.training:
            update = update.detach()

        # Apply the update to the iterate
        next_step = update * self.step_size

        return next_step