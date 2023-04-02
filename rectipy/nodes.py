"""
Contains the network classes, which are the main interfaces for the user to interact with.
"""

from pyrates import NodeTemplate, CircuitTemplate, clear, clear_frontend_caches
import torch
from torch.nn import Module, Tanh, Softmax, Softmin, Sigmoid, Identity
from typing import Callable, Union, Iterator
import numpy as np
from .utility import to_device


class ActivationFunction:

    def __init__(self, n: int, func: str, **kwargs):

        if func == 'tanh':
            func = Tanh
        elif func == 'softmax':
            func = Softmax
            if "dim" not in kwargs:
                kwargs["dim"] = 0
        elif func == 'softmin':
            func = Softmin
            if "dim" not in kwargs:
                kwargs["dim"] = 0
        elif func == 'sigmoid':
            func = Sigmoid
        elif func == "identity":
            func = Identity
        else:
            raise ValueError(f"Invalid keyword argument `func`: {func} is not a valid "
                             f"option. See the docstring for `Network.add_ffwd_layer` for valid options.")

        self.n_in = n
        self.n_out = n
        self.func = func(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.func.forward(x)

    def parameters(self, **kwargs):
        return self.func.parameters(**kwargs)


class RateNet(Module):

    def __init__(self, rnn_func: Callable, rnn_args: tuple, var_map: dict, param_map: dict, dt: float = 1e-3,
                 dtype: torch.dtype = torch.float64, train_params: list = None, device: str = "cpu", **kwargs):

        super().__init__()

        # private attributes
        self._args = [to_device(arg, device) for arg in rnn_args[1:]]
        self._var_map = var_map
        self._param_map = param_map
        self._start = torch.tensor(self._var_map["out"][0], dtype=torch.int64, device=device)
        self._stop = torch.tensor(self._var_map["out"][-1], dtype=torch.int64, device=device)
        self._inp_ext = self._param_map["in"]

        # public attributes
        self.dt = dt
        self.func = rnn_func
        self.device = device
        self.n_out = int((self._stop - self._start).detach().cpu().numpy())
        self.n_in = int(self._args[self._inp_ext].shape[0])

        # initialize trainable parameters
        self.train_params = [self._args[self._param_map[p]] for p in train_params] if train_params else []
        for p in self.train_params:
            p.requires_grad = True

        # initialize state vector
        require_grad = True if len(self.train_params) >= 1 else False
        self.y = torch.tensor(rnn_args[0].detach().numpy(), dtype=dtype, requires_grad=require_grad, device=device)

    def __getitem__(self, item):
        try:
            return self._args[self._param_map[item]]
        except KeyError:
            idx = self._var_map[item]
            if type(idx) is tuple:
                return self.y[idx[0]:idx[1]]
            return self.y[idx]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @property
    def parameter_names(self) -> list:
        return list(self._param_map.keys())

    @property
    def variable_names(self) -> list:
        return list(self._var_map.keys())

    @classmethod
    def from_yaml(cls, node: Union[str, NodeTemplate], input_var: str, output_var: str, weights: np.ndarray = None,
                  source_var: str = None, target_var: str = None, train_params: list = None, **kwargs):

        # extract keyword arguments for initialization
        dt = kwargs.pop('dt', 1e-3)
        clear_template = kwargs.pop('clear', True)
        dtype = kwargs.pop('dtype', torch.float64)
        kwargs['float_precision'] = str(dtype).split('.')[-1]
        param_mapping = kwargs.pop("param_mapping", {})
        param_mapping["in"] = input_var
        param_mapping["weights"] = "in_edge_0/weight"
        var_mapping = kwargs.pop("var_mapping", {})
        var_mapping["out"] = output_var

        # generate rnn template and function
        try:
            func, args, keys, template, var_map = \
                cls._circuit_from_yaml(node, dt, weights=weights, source_var=source_var, target_var=target_var,
                                       **kwargs)
        except Exception as e:
            clear_frontend_caches()
            raise e

        # get parameter and variable indices
        param_map = cls._get_param_indices(template, keys[1:])
        param_map = _remove_node_from_dict_keys(param_map)
        for key, var in param_mapping.items():
            param_map[key] = param_map[var]
        var_map.update(cls._get_var_indices(template, var_mapping))
        var_map = _remove_node_from_dict_keys(var_map)

        if clear_template:
            clear(template)
        return cls(func, args, var_map, param_map, dt=dt, train_params=train_params, dtype=dtype, **kwargs)

    @classmethod
    def from_template(cls, template: Union[str, CircuitTemplate], input_var: str, output_var: str,
                      train_params: list = None, device: str = "cpu", **kwargs):

        # extract keyword arguments for initialization
        dt = kwargs.pop('dt', 1e-3)
        clear_template = kwargs.pop('clear', True)
        dtype = kwargs.pop('dtype', torch.float64)
        kwargs['float_precision'] = str(dtype).split('.')[-1]
        param_mapping = kwargs.pop("param_mapping", {})
        var_map = kwargs.pop("var_mapping", {})
        var_map["out"] = output_var

        # generate rnn template and function
        try:
            func, args, keys, state_var_indices = template.get_run_func('rnn_layer', dt, backend='torch', clear=False,
                                                                        inplace_vectorfield=False, **kwargs)
        except Exception as e:
            clear_frontend_caches()
            raise e

        # get parameter and variable indices
        param_map = cls._get_param_indices(template, keys[1:])
        param_map = _remove_node_from_dict_keys(param_map)
        for key, var in param_mapping.items():
            param_map[key] = param_map[var]
        var_map.update(cls._get_var_indices(template, var_map))
        var_map = _remove_node_from_dict_keys(var_map)

        if clear_template:
            clear(template)
        return cls(func, args, var_map, param_map, dt=dt, train_params=train_params, dtype=dtype, device=device,
                   **kwargs)

    def forward(self, x):
        self._args[self._inp_ext] = x
        y_old = self.y
        dy = self.func(0, y_old, *self._args)
        self.y = y_old + self.dt * dy
        return y_old[self._start:self._stop]

    def parameters(self, recurse: bool = True) -> Iterator:
        for p in self.train_params:
            yield p

    def detach(self):
        self.y = self.y.detach()
        self._args = [arg.detach() if type(arg) is torch.Tensor else arg for arg in self._args]

    def reset(self, y: np.ndarray, idx: np.ndarray = None):
        # TODO: remove this method?
        if idx is None:
            self.y = torch.tensor(y, dtype=self.y.dtype, device=self.device)
        else:
            y_new = self.y.clone()
            y_new[torch.tensor(idx, dtype=torch.long)] = torch.tensor(y, dtype=y_new.dtype)
            self.y = to_device(y_new, self.device)

    @classmethod
    def _circuit_from_yaml(cls, node: Union[str, NodeTemplate], dt: float, weights: np.ndarray = None, source_var: str = None,
                           target_var: str = None, **kwargs) -> tuple:

        # initialize base node template
        if type(node) is str:
            node = NodeTemplate.from_yaml(node)

        # initialize base circuit template
        n = kwargs.pop("N") if weights is None else weights.shape[0]
        nodes = {f'n{i}': node for i in range(n)}
        template = CircuitTemplate(name='reservoir', nodes=nodes)

        # add edges to network
        if weights is not None:
            if source_var is None or target_var is None:
                raise ValueError("If synaptic weights are passed (`weights`), please provide the names of the source "
                                 "and target variable that should be connected via `weights`.")
            edge_attr = kwargs.pop("edge_attr", None)
            template.add_edges_from_matrix(source_var, target_var, nodes=list(nodes.keys()), weight=weights,
                                           edge_attr=edge_attr)

        # add variable updates
        if 'node_vars' in kwargs:
            template.update_var(node_vars=kwargs.pop('node_vars'))

        # generate rnn function
        func, args, keys, state_var_indices = template.get_run_func('rnn_layer', dt, backend='torch', clear=False,
                                                                    inplace_vectorfield=False, **kwargs)

        return func, args[1:], keys[1:], template, state_var_indices

    @staticmethod
    def _get_var_indices(template: CircuitTemplate, variables: dict):
        var_dict = {key: f"all/{val}" for key, val in variables.items()}
        var_indices, _ = template.get_variable_positions(var_dict)
        results = {}
        for var in list(var_indices.keys()):
            try:
                results[var] = [val[0] for val in var_indices.pop(var).values()]
            except AttributeError:
                results[var] = var_indices.pop(var)
            results[var] = (results[var][0], results[var][-1] + 1)
        return results

    @staticmethod
    def _get_param_indices(template: CircuitTemplate, params: tuple) -> dict:
        param_mapping = {}
        for p in params:
            try:
                p_tmp, _ = template.get_var(p)
                if hasattr(p_tmp, 'name'):
                    p_tmp = p_tmp.name
                idx = params.index(p_tmp)
            except (IndexError, ValueError):
                idx = params.index(p)
            param_mapping[p] = idx
        return param_mapping


class SpikeNet(RateNet):

    def __init__(self, rnn_func: Callable, rnn_args: tuple, var_map: dict, param_map: dict,
                 spike_threshold: float = 1e2, spike_reset: float = -1e2, dt: float = 1e-3,
                 dtype: torch.dtype = torch.float64, train_params: list = None, device: str = None, **kwargs):

        super().__init__(rnn_func, rnn_args, var_map, param_map, dt=dt, dtype=dtype, train_params=train_params,
                         device=device)
        self._spike_var = self._param_map['spike_var']
        self._thresh = spike_threshold
        self._reset = spike_reset
        self._spike_start = torch.tensor(self._var_map['spike_def'][0], dtype=torch.int64, device=self.device)
        self._spike_stop = torch.tensor(self._var_map['spike_def'][-1], dtype=torch.int64, device=self.device)

    @classmethod
    def from_yaml(cls, node: Union[str, NodeTemplate], input_var: str, output_var: str, weights: np.ndarray = None,
                  source_var: str = None, target_var: str = None, spike_var: str = 'spike', spike_def: str = 'v',
                  train_params: list = None, **kwargs):

        # extract keyword arguments for initialization
        kwargs["param_mapping"] = {"spike_var": spike_var}
        kwargs["var_mapping"] = {"spike_def": spike_def}

        return super().from_yaml(node, input_var, output_var, weights, source_var, target_var,
                                 train_params=train_params, **kwargs)

    @classmethod
    def from_template(cls, template: Union[str, CircuitTemplate], input_var: str, output_var: str,
                      spike_var: str = 'spike', spike_def: str = 'v', train_params: list = None, **kwargs):
        # extract keyword arguments for initialization
        kwargs["param_mapping"] = {"spike_var": spike_var}
        kwargs["var_mapping"] = {"spike_def": spike_def}

        return super().from_template(template, input_var, output_var, train_params=train_params, **kwargs)

    def forward(self, x):
        spikes = self.y[self._spike_start:self._spike_stop] >= self._thresh
        self._args[self._spike_var] = spikes / self.dt
        self._args[self._inp_ext] = x
        y_old = self.y
        dy = self.func(0, y_old, *self._args)
        self.y = y_old + self.dt * dy
        self.spike_reset(spikes)
        return y_old[self._start:self._stop]

    def spike_reset(self, spikes: torch.Tensor):
        self.y[self._spike_start:self._spike_stop][spikes] = self._reset


def _remove_node_from_dict_keys(mapping: dict) -> dict:
    new_mapping = dict()
    for key, val in mapping.items():
        try:
            *node, op, var = key.split("/")
            new_mapping[f"{op}/{var}"] = val
        except ValueError:
            new_mapping[key] = val
    return new_mapping
