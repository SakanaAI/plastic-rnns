import torch
from torch import nn

class RecurrentModel(nn.Module):               
    def __init__(self, d_input, d_model, cell_cls, model_kwargs):
        super(RecurrentModel, self).__init__()
        self.d_model = d_model
        self.recurrent_module = cell_cls(d_input=d_input, d_model=d_model, **model_kwargs)

    def get_initial_state(self):
        return self.recurrent_module.initialize_state()

    def forward(self, input, state):
        assert input.ndim == 3, "Input should be of size T x B x D"
        outputs = []
        for element in input:
            hidden_state, state = self.recurrent_module(element, state)
            outputs.append(hidden_state)
        return torch.stack(outputs, dim=0), state
