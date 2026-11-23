import copy
from omegaconf import DictConfig
from torch import nn

from models.base import RecurrentModel
from models.cells.ctm import ContinuousThoughtMachineCell as CTMCell
from models.cells.ctm import ContinuousThoughtMachineCellNoNLMs as CTMCellNoNLMs
from models.cells.ctm import ContinuousThoughtMachineHebbianCell
from models.cells.lstm import LSTMCell, PlasticLSTMCell, HyperLSTMCell
from models.cells.rnn import RNNCell, PlasticRNNCell
from models.cells.stpn import STPNCell as STPN
from models.cells.fw import FW as FW
from models.cells.ctm import CrossAttentionCTMCell
from models.cells.gated_delta_net import GatedDeltaNetCell

_CELLS = {
	"CTM": CTMCell,
	"PlasticCTMNoDynamicRescaling": CTMCell,
	"LSTM": LSTMCell,
	"PlasticLSTM": PlasticLSTMCell,
	"HyperLSTM": HyperLSTMCell,
	"RNN": RNNCell,
	"PlasticRNN": PlasticRNNCell,
	"STPN": STPN,
	"CTMNoNLMs": CTMCellNoNLMs,
	"CTMHebbian": ContinuousThoughtMachineHebbianCell,
	"FW": FW,
	"CrossAttentionCTM": CrossAttentionCTMCell,
	"GDN": GatedDeltaNetCell,
}

def construct_rnn(d_input, d_model, model_type, model_kwargs: DictConfig) -> nn.Module:
	model_cfg_copy = copy.deepcopy(model_kwargs)
	del model_cfg_copy.d_input, model_cfg_copy.d_model
	return RecurrentModel(d_input=d_input, d_model=d_model, cell_cls=_CELLS[model_type], model_kwargs=model_cfg_copy)
