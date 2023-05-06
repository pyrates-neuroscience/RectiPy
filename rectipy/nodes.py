"""
Contains the network classes, which are the main interfaces for the user to interact with.
"""

from pyrates import NodeTemplate, CircuitTemplate, clear, clear_frontend_caches
import torch
from torch.nn import Module, Tanh, Softmax, Softmin, Sigmoid, Identity, LogSoftmax
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
        elif func == "log_softmax":
            func = LogSoftmax
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

    state_vars = ["y"]

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
        in_arg = self._args[self._inp_ext]
        self.n_in = int(in_arg.shape[0]) if hasattr(in_arg, "shape") else 1

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
    def from_pyrates(cls, node: Union[str, NodeTemplate, CircuitTemplate], input_var: str, output_var: str,
                     weights: np.ndarray = None, source_var: str = None, target_var: str = None,
                     train_params: list = None, **kwargs):

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
            if isinstance(node, CircuitTemplate):
                template = node
                var_map = var_mapping
                func, args, keys, state_var_indices = node.get_run_func('rnn_layer', dt, backend='torch', clear=False,
                                                                        inplace_vectorfield=False, **kwargs)
            else:
                func, args, keys, template, var_map = cls._circuit_from_yaml(node, dt, weights=weights,
                                                                             source_var=source_var,
                                                                             target_var=target_var, **kwargs)
        except Exception as e:
            clear_frontend_caches()
            raise e

        # make sure the state vector is the first argument in args and keys
        start = keys.index("y")
        args = args[start:]
        keys = keys[start:]

        # get parameter and variable indices
        param_map = cls._get_param_indices(template, keys[1:])
        param_map = _remove_node_from_dict_keys(param_map)
        for key, var in param_mapping.items():
            try:
                param_map[key] = param_map[var]
            except KeyError:
                pass
        var_map.update(cls._get_var_indices(template, var_mapping))
        var_map = _remove_node_from_dict_keys(var_map)

        # clean up and return an instance of the class
        if clear_template:
            clear(template)
        return cls(func, args, var_map, param_map, dt=dt, train_params=train_params, dtype=dtype, **kwargs)

    def forward(self, x):
        self._args[self._inp_ext] = x
        y_old = self.y
        self.y = y_old + self.dt * self.func(0, y_old, *self._args)
        return y_old[self._start:self._stop]

    def parameters(self, recurse: bool = True) -> Iterator:
        for p in self.train_params:
            yield p

    def detach(self, requires_grad: bool = False, detach_params: bool = True):
        set_grad = []
        for key in self.state_vars:
            v = getattr(self, key)
            v_new = v.detach()
            setattr(self, key, v_new)
            set_grad.append(v_new)
        if detach_params:
            self._args = [arg.detach() if type(arg) is torch.Tensor else arg for arg in self._args]
            set_grad.extend(self._args)
        for v in set_grad:
            if type(v) is torch.Tensor:
                v.requires_grad = requires_grad

    def reset(self, y: Union[np.ndarray, torch.Tensor] = None, idx: np.ndarray = None):
        if y is None:
            y = np.zeros_like(self.y.detach().cpu().numpy())
        if type(y) is torch.Tensor:
            y = y.clone().detach()
            y.requires_grad = self.y.requires_grad
        else:
            y = torch.tensor(y, dtype=self.y.dtype, requires_grad=self.y.requires_grad)
        if idx is None:
            self.y = y
        else:
            y_new = self.y.clone().detach()
            y_new[torch.tensor(idx, dtype=torch.long)] = y
            self.y = to_device(y_new, self.device)

    def set_param(self, param: str, val: Union[torch.Tensor, float]):
        """Set value of a node parameter.

        Parameters
        ----------
        param
            Name of the parameter.
        val
            New value of that parameter.

        Returns
        -------
        None
        """
        try:
            self._args[self._param_map[param]] = val
        except KeyError:
            raise KeyError(f"Parameter {param} was not found on the node.")

    @classmethod
    def _circuit_from_yaml(cls, node: Union[str, NodeTemplate], dt: float, weights: np.ndarray = None,
                           source_var: str = None, target_var: str = None, **kwargs) -> tuple:

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
            template.add_edges_from_matrix(source_var, target_var, source_nodes=list(nodes.keys()), weight=weights,
                                           edge_attr=edge_attr)

        # add variable updates
        if 'node_vars' in kwargs:
            template.update_var(node_vars=kwargs.pop('node_vars'))

        # generate rnn function
        func, args, keys, state_var_indices = template.get_run_func('rnn_layer', dt, backend='torch', clear=False,
                                                                    inplace_vectorfield=False, **kwargs)

        return func, args, keys, template, state_var_indices

    @staticmethod
    def _get_var_indices(template: CircuitTemplate, variables: dict):
        var_dict = {key: f"all/{val}" if "all/" not in val else val for key, val in variables.items()}
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

    state_vars = ["_y_start", "_y_spike", "_y_stop"]

    def __init__(self, rnn_func: Callable, rnn_args: tuple, var_map: dict, param_map: dict,
                 spike_threshold: float = 1e2, spike_reset: float = -1e2, dt: float = 1e-3,
                 dtype: torch.dtype = torch.float64, train_params: list = None, device: str = None, **kwargs):

        super().__init__(rnn_func, rnn_args, var_map, param_map, dt=dt, dtype=dtype, train_params=train_params,
                         device=device, **kwargs)

        # define spiking function
        Spike.center = torch.tensor(kwargs.pop("spike_center", 1.0), device=self.device, dtype=self.y.dtype)
        Spike.slope = torch.tensor(kwargs.pop("spike_slope", 200.0/np.abs(spike_threshold - spike_reset)),
                                   device=self.device, dtype=self.y.dtype)
        self.spike = Spike.apply

        # set private attributes
        self._spike_var = self._param_map['spike_var']
        self._reset = torch.tensor(spike_reset, device=self.device, dtype=self.y.dtype)
        self._thresh = torch.tensor(spike_threshold, device=self.device, dtype=self.y.dtype)

        # define state variable slices for updates
        self._spike_start = torch.tensor(self._var_map['spike_def'][0], dtype=torch.int64, device=self.device)
        self._spike_stop = torch.tensor(self._var_map['spike_def'][-1], dtype=torch.int64, device=self.device)
        self._y_start = torch.tensor(0.0)
        self._y_spike = torch.tensor(0.0)
        self._y_stop = torch.tensor(0.0)
        self._init_state()

    @classmethod
    def from_pyrates(cls, node: Union[str, NodeTemplate, CircuitTemplate], input_var: str, output_var: str,
                     weights: np.ndarray = None, source_var: str = None, target_var: str = None,
                     spike_var: str = 'spike', spike_def: str = 'v', train_params: list = None, **kwargs):

        # extract keyword arguments for initialization
        kwargs["param_mapping"] = {"spike_var": spike_var}
        kwargs["var_mapping"] = {"spike_def": spike_def}

        return super().from_pyrates(node, input_var, output_var, weights, source_var, target_var,
                                    train_params=train_params, **kwargs)

    def forward(self, x):
        spikes = self.spike(self._y_spike - self._thresh)
        reset = spikes.detach()
        self._args[self._spike_var] = spikes / self.dt
        self._args[self._inp_ext] = x
        self.y = torch.cat((self._y_start, self._y_spike, self._y_stop), 0)
        y_new = self.y + self.dt * self.func(0, self.y, *self._args)
        self._y_start = y_new[:self._spike_start]
        self._y_spike = y_new[self._spike_start:self._spike_stop]*(1.0-reset) + reset*self._reset
        self._y_stop = y_new[self._spike_stop:]
        return self.y[self._start:self._stop]

    def reset(self, y: np.ndarray = None, idx: np.ndarray = None):
        super().reset(y=y, idx=idx)
        self._init_state()

    def _init_state(self):
        self._y_start = self.y[:self._spike_start].clone()
        self._y_spike = self.y[self._spike_start:self._spike_stop].clone()
        self._y_stop = self.y[self._spike_stop:].clone()


class Spike(torch.autograd.Function):

    slope = 10.0
    center = torch.tensor(1.0)

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return torch.heaviside(x, Spike.center)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x, = ctx.saved_tensors
        return grad_output/(1.0 + Spike.slope*torch.abs(x))**2


def _remove_node_from_dict_keys(mapping: dict) -> dict:
    new_mapping = dict()
    for key, val in mapping.items():
        try:
            *node, op, var = key.split("/")
            new_mapping[f"{op}/{var}"] = val
        except ValueError:
            new_mapping[key] = val
    return new_mapping
