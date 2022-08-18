from pyrates import NodeTemplate, CircuitTemplate, clear
import torch
from torch.nn import Module
from typing import Callable, Union, Iterator
import numpy as np


class RNNLayer(Module):

    def __init__(self, rnn_func: Callable, rnn_args: tuple, input_ext: int, output: list, dt: float = 1e-3,
                 dtype: torch.dtype = torch.float64, train_params: list = None, record_vars: dict = None):

        super().__init__()
        self.y = torch.tensor(rnn_args[0].detach().numpy(), dtype=dtype)
        self.dy = torch.zeros_like(self.y)
        self.output = torch.tensor(output, dtype=torch.int64)
        self.dt = dt
        self.func = rnn_func
        self.args = list(rnn_args[1:])
        self.train_params = [self.args[idx-1] for idx in train_params] if train_params else []
        self._record_vars = record_vars
        self._inp_ext = input_ext - 1

    @classmethod
    def from_yaml(cls, node: Union[str, NodeTemplate], weights: np.ndarray, source_var: str, target_var: str,
                  input_var: str, output_var: str, train_params: list = None, record_vars: list = None, **kwargs):

        # extract keyword arguments for initialization
        dt = kwargs.pop('dt', 1e-3)
        clear_template = kwargs.pop('clear', True)

        # generate rnn template and function
        func, args, keys, template, state_var_indices = cls._circuit_from_yaml(node, weights, source_var, target_var,
                                                                               step_size=dt, **kwargs)

        # get variable indices
        input_idx = cls._get_param_indices(template, [input_var], keys)[0]
        if train_params:
            train_params = cls._get_param_indices(template, train_params, keys)
        var_indices = cls._get_var_indices(template, output_var, recording_vars=record_vars)

        if clear_template:
            clear(template)
        return cls(func, args, input_idx, var_indices.pop('out'), dt=dt, train_params=train_params,
                   record_vars=var_indices)

    def forward(self, x):
        self.args[self._inp_ext] = x
        self.dy = self.func(0, self.y.detach(), *self.args)
        self.y = self.y + self.dt * self.dy
        return self.y[self.output]

    def record(self, variables: list):
        for v in variables:
            yield self.y[self._record_vars[v]]

    def parameters(self, recurse: bool = True) -> Iterator:
        for p in self.train_params:
            yield p

    def detach(self):
        self.y = self.y.detach()
        self.dy = self.dy.detach()
        self.args = [arg.detach() for arg in self.args]

    @classmethod
    def _circuit_from_yaml(cls, node: Union[str, NodeTemplate], weights: np.ndarray, source_var: str, target_var: str,
                           **kwargs) -> tuple:

        # initialize base node template
        if type(node) is str:
            node = NodeTemplate.from_yaml(node)

        # initialize base circuit template
        n = weights.shape[0]
        nodes = {f'n{i}': node for i in range(n)}
        template = CircuitTemplate(name='reservoir', nodes=nodes)

        # add edges to network
        edge_attr = kwargs.pop("edge_attr", None)
        template.add_edges_from_matrix(source_var, target_var, nodes=list(nodes.keys()), weight=weights,
                                       edge_attr=edge_attr)

        # add variable updates
        if 'node_vars' in kwargs:
            template.update_var(node_vars=kwargs.pop('node_vars', None))

        # generate rnn function
        func, args, keys, state_var_indices = template.get_run_func('rnn_layer', backend='torch', clear=False,
                                                                    inplace_vectorfield=False, **kwargs)

        return func, args[1:], keys[1:], template, state_var_indices

    @staticmethod
    def _get_var_indices(template: CircuitTemplate, out_var: str, spike_var: str = None, recording_vars: list = None):
        var_dict = {'out': f"all/{out_var}"}
        if spike_var is not None:
            var_dict['spike'] = f"all/{spike_var}"
        if recording_vars:
            var_dict.update({key: f"all/{key}" for key in recording_vars})
        var_indices, _ = template.get_variable_positions(var_dict)
        results = {}
        for var in list(var_indices.keys()):
            try:
                results[var] = [val[0] for val in var_indices.pop(var).values()]
            except AttributeError:
                results[var] = var_indices.pop(var)
        return results

    @staticmethod
    def _get_param_indices(template: CircuitTemplate, target_params: list, all_params: tuple, node_key: str = "n0"):
        indices = []
        for p in target_params:
            try:
                p_tmp, _ = template.get_var(f"{node_key}/{p}")
                if hasattr(p_tmp, 'name'):
                    p_tmp = p_tmp.name
                idx = all_params.index(p_tmp)
            except IndexError:
                idx = all_params.index(p)
            indices.append(idx)
        return indices


class SRNNLayer(RNNLayer):

    def __init__(self, rnn_func: Callable, rnn_args: tuple, input_ext: int, input_net: int, output: list,
                 spike_var: list, spike_threshold: float = 1e2, spike_reset: float = -1e2, dt: float = 1e-3,
                 dtype: torch.dtype = torch.float64, train_params: list = None, record_vars: dict = None):

        super().__init__(rnn_func, rnn_args, input_ext, output, dt=dt, dtype=dtype, train_params=train_params,
                         record_vars=record_vars)
        self._inp_net = input_net - 2
        self._thresh = spike_threshold
        self._reset = spike_reset
        self._var = torch.tensor(spike_var, dtype=torch.int64)

    @classmethod
    def from_yaml(cls, node: Union[str, NodeTemplate], weights: np.ndarray, source_var: str, target_var: str,
                  input_var_ext: str, input_var_net: str, output_var: str, spike_var: str, train_params: list = None,
                  record_vars: list = None, **kwargs):

        # extract keyword arguments for initialization
        dt = kwargs.pop('dt', 1e-3)
        kwargs_init = {}
        for key in ['spike_threshold', 'spike_reset']:
            if key in kwargs:
                kwargs_init[key] = kwargs.pop(key)
        clear_template = kwargs.pop('clear', True)

        # generate rnn template and function
        func, args, keys, template, _ = cls._circuit_from_yaml(node, weights, source_var, target_var, step_size=dt,
                                                               **kwargs)

        # get variable indices
        input_ext_idx, input_net_idx = cls._get_param_indices(template, [input_var_ext, input_var_net], keys)
        if train_params:
            train_params = cls._get_param_indices(template, train_params, keys)
        var_indices = cls._get_var_indices(template, output_var, spike_var=spike_var, recording_vars=record_vars)

        if clear_template:
            clear(template)
        return cls(func, args, input_ext_idx, input_net_idx, var_indices.pop('out'), var_indices.pop('spike'), dt=dt,
                   train_params=train_params, record_vars=var_indices, **kwargs_init)

    def forward(self, x):
        spikes = self.y[self._var] >= self._thresh
        self.args[self._inp_net] = spikes/self.dt
        self.args[self._inp_ext] = x
        self.dy = self.func(0, self.y, *self.args)
        self.y = self.y + self.dt * self.dy
        self.reset(spikes)
        return self.y[self.output]

    def reset(self, spikes: torch.Tensor):
        self.y[self._var[spikes]] = self._reset
