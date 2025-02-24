## Modified from https://github.com/xhchrn/MS4L2O 

import math
import torch
import numpy               as np
import torch.nn            as nn
import torch.optim         as optim
import torch.nn.functional as F

#from optimizees.base import BaseOptimizee

ESP = 1e-8

NORM_FUNC = {
    'exp': torch.exp,
    'eye': nn.Identity(),
    'sigmoid': lambda x: 2.0 * torch.sigmoid(x),
    'softplus': nn.Softplus(),
}


class RNNprop(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, layers,
                 beta1, beta2, **kwargs):
        """
        An implement of the RNNprop model proposed in:
		Lv et al. (2017) "Learning Gradient Descent: Better Generalization and Longer Horizons."
        """
        super().__init__()

        self.input_size  = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        use_bias = True
        
        self.beta1, self.beta2 = beta1, beta2

        self.layers = layers  # Number of layers for LSTM

        self.elu = nn.ELU()
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(hidden_size, hidden_size, layers, bias=use_bias)

        self.linear_in = nn.Linear(self.input_size, hidden_size, bias=use_bias)
        self.linear_out = nn.Linear(hidden_size, output_size, bias=True)

        self.state = None
        self.step_size = kwargs.get('step_size', None)

    @property
    def device(self):
        return self.linear_in.weight.device


    def reset_state(self, P: int, step_size: float, **kwargs):
        scale = kwargs.get('state_scale', 0.01)
        batch_size = P
        self.state = (
            # hidden_state
            scale * torch.randn(
                self.layers, batch_size, self.hidden_size
            ).to(self.device),
            # cell_state
            scale * torch.randn(
                self.layers, batch_size, self.hidden_size
            ).to(self.device),
        )
        self.step_size = step_size
        
        self.m = torch.zeros(P)
        self.v = torch.zeros(P)
        
        self.b1t = 1
        self.b2t = 1

    # def detach_state(self):
    #     if self.state is not None:
    #         self.state = (self.state[0].detach(), self.state[1].detach())

    def name(self):
        """
        Function: name
        Purpose : Identify the name of the model.
        """
        return 'RNNprop'

    def forward(self, grad, 
                reset_state: bool = False, 
    ):
        
        batch_size = grad.shape[0]
        
            
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        self.b1t *= self.beta1
        self.b2t *= self.beta2
        
        sv = torch.sqrt(self.v / (1 - self.b2t)) + ESP

        lstm_input = grad / sv
        lstm_input2 = self.m / (1 - self.b1t) / sv

        # Here the `grad` is of dimension (batch_size, input_size, 1). Need
        # to reshape it into (1, batch_size, input_size) to be used by LSTM.
        # lstm_input = lstm_input.squeeze().unsqueeze(0)
        lstm_input = lstm_input.flatten().unsqueeze(0).unsqueeze(-1)
        lstm_input2 = lstm_input2.flatten().unsqueeze(0).unsqueeze(-1)
        lstm_in = torch.cat((lstm_input,lstm_input2), dim = 2)
        
        output = self.elu(self.linear_in(lstm_in))
        # Core update by LSTM.
        output, self.state = self.lstm(output, self.state)
        output = self.tanh(self.linear_out(output)).reshape_as(grad)
        if not self.training:
            output = output.detach()

        next_step = - output * self.step_size

        return next_step