import torch
from torch.nn import Sequential
from typing import Union, Iterator, Callable
from .rnn_layer import RNNLayer, SRNNLayer
from .input_layer import InputLayer
from .output_layer import OutputLayer
from .utility import retrieve_from_dict, add_op_name
from .observer import Observer
from pyrates import NodeTemplate, clear
import numpy as np
from time import perf_counter


class Network:

    def __init__(self, rnn_layer: Union[RNNLayer, SRNNLayer], var_map: dict = None):

        self.rnn_layer = rnn_layer
        self.input_layer = None
        self.output_layer = None
        self._var_map = var_map if var_map else {}

    @classmethod
    def from_yaml(cls, node: Union[str, NodeTemplate], weights: np.ndarray, source_var: str, target_var: str,
                  input_var_ext: str, output_var: str, input_var_net: str = None, spike_var: str = None, op: str = None,
                  train_params: list = None, record_vars: list = None, **kwargs):

        # add operator key to variable names
        var_dict = {'svar': source_var, 'tvar': target_var, 'in_ext': input_var_ext, 'in_net': input_var_net,
                    'out': output_var, 'spike': spike_var}
        new_vars = {}
        if op is not None:
            for key, var in var_dict.copy().items():
                var_dict[key] = add_op_name(op, var, new_vars)
            if train_params:
                train_params = [add_op_name(op, p, new_vars) for p in train_params]
            if record_vars:
                record_vars = [add_op_name(op, v, new_vars) for v in record_vars]

        # initialize rnn layer
        if input_var_net is None and spike_var is None:
            rnn_layer = RNNLayer.from_yaml(node, weights, var_dict['svar'], var_dict['tvar'], var_dict['in_ext'],
                                           var_dict['out'], train_params=train_params, record_vars=record_vars,
                                           **kwargs)
        elif input_var_net is None or spike_var is None:
            raise ValueError('To define a reservoir with a spiking neural network layer, please provide both the '
                             'name of the variable that spikes should be stored in (`input_var_net`) as well as the '
                             'name of the variable that is used to define spikes (`spike_var`).')
        else:
            rnn_layer = SRNNLayer.from_yaml(node, weights, var_dict['svar'], var_dict['tvar'], var_dict['in_ext'],
                                            var_dict['in_net'], var_dict['out'], var_dict['spike'],
                                            train_params=train_params, record_vars=record_vars, **kwargs)

        # initialize model
        return cls(rnn_layer, var_map=new_vars)

    def add_input_layer(self, n: int, m: int, weights: torch.Tensor = None, trainable: bool = False,
                        dtype: torch.dtype = torch.float64) -> InputLayer:

        # initialize input layer
        input_layer = InputLayer(n, m, weights, trainable=trainable, dtype=dtype)

        # add layer to model
        self.input_layer = input_layer

        # return layer
        return self.input_layer

    def add_output_layer(self, n: int, m: int, weights: torch.Tensor = None, trainable: bool = False,
                         transform: str = None, dtype: torch.dtype = torch.float64) -> OutputLayer:

        # initialize output layer
        output_layer = OutputLayer(n, m, weights, trainable=trainable, transform=transform, dtype=dtype)

        # add layer to model
        self.output_layer = output_layer

        # return layer
        return self.output_layer

    def remove_input_layer(self):
        self.input_layer = None

    def remove_output_layer(self):
        self.output_layer = None

    def train(self, inputs: np.ndarray, targets: np.ndarray, optimizer: str = 'sgd', optimizer_kwargs: dict = None,
              loss: str = 'mse', loss_kwargs: dict = None, lr: float = 1e-3, device: str = None,
              sampling_steps: int = 100, verbose: bool = True, **kwargs) -> Observer:

        # preparations
        ##############

        # transform inputs into tensors
        inp_tensor = torch.tensor(inputs)
        target_tensor = torch.tensor(targets)
        if inp_tensor.shape[0] != target_tensor.shape[0]:
            raise ValueError('Wrong dimensions of input and target output. Please make shure that `inputs` and '
                             '`targets` agree in the first dimension.')

        # set up model
        if not list(self.rnn_layer.parameters()):
            self.rnn_layer.detach()
        model = self._compile(device)

        # initialize loss function
        loss = self._get_loss_function(loss, loss_kwargs=loss_kwargs)

        # initialize optimizer
        optimizer = self._get_optimizer(optimizer, lr, model.parameters(), optimizer_kwargs=optimizer_kwargs)

        # retrieve keyword arguments for optimization
        step_kwargs = retrieve_from_dict(['closure'], kwargs)
        error_kwargs = retrieve_from_dict(['retain_graph'], kwargs)

        # initialize observer
        obs_kwargs = retrieve_from_dict(['record_output', 'record_loss', 'record_vars'], kwargs)
        obs = Observer(dt=self.rnn_layer.dt, **obs_kwargs)
        rec_vars = [self._relabel_var(v) for v in obs.recorded_state_variables]

        # optimization
        ##############

        steps = inp_tensor.shape[0]
        t0 = perf_counter()
        for step in range(steps):

            # forward pass
            prediction = model(inp_tensor[step, :])

            # loss calculation
            error = loss(prediction, target_tensor[step, :])

            # error backpropagation
            optimizer.zero_grad()
            error.backward(**error_kwargs)
            optimizer.step(**step_kwargs)

            # results storage
            if step % sampling_steps == 0:
                if verbose:
                    print(f'Progress: {step}/{steps} training steps finished.')
                obs.record(prediction, error.item(), self.rnn_layer.record(rec_vars))

        t1 = perf_counter()
        print(f'Finished optimization after {t1-t0} s.')
        return obs

    def test(self, inputs: np.ndarray, targets: np.ndarray, loss: str = 'mse', loss_kwargs: dict = None,
             device: str = None, sampling_steps: int = 100, verbose: bool = True, **kwargs) -> tuple:

        # preparations
        ##############

        # transform inputs into tensors
        inp_tensor = torch.tensor(inputs)
        target_tensor = torch.tensor(targets)
        if inp_tensor.shape[0] != target_tensor.shape[0]:
            raise ValueError('Wrong dimensions of input and target output. Please make shure that `inputs` and '
                             '`targets` agree in the first dimension.')

        # set up model
        model = self._compile(device)

        # initialize loss function
        loss = self._get_loss_function(loss, loss_kwargs=loss_kwargs)

        # initialize observer
        obs_kwargs = retrieve_from_dict(['record_output', 'record_loss', 'record_vars'], **kwargs)
        obs = Observer(dt=self.rnn_layer.dt, **obs_kwargs)
        rec_vars = [self._relabel_var(v) for v in obs.recorded_state_variables]

        # test loop
        ###########

        loss_total = 0.0
        steps = inp_tensor.shape[0]
        with torch.no_grad():
            for step in range(steps):

                # forward pass
                prediction = model(inp_tensor[step, :])

                # loss calculation
                error = loss(prediction, target_tensor[step, :])
                loss_total += error.item()

                # results storage
                if step % sampling_steps == 0:
                    if verbose:
                        print(f'Progress: {step}/{steps} test steps finished.')
                    obs.record(prediction, error.item(), self.rnn_layer.record(rec_vars))

        return obs, loss_total

    def run(self, inputs: np.ndarray, device: str = 'cpu', sampling_steps: int = 1, verbose: bool = True, **kwargs
            ) -> Observer:

        # transform input into tensor
        steps = inputs.shape[0]
        inp_tensor = torch.tensor(inputs)

        # initialize model from layers
        model = self._compile(device)

        # initialize observer
        obs = Observer(dt=self.rnn_layer.dt, record_loss=False, **kwargs)
        rec_vars = [self._relabel_var(v) for v in obs.recorded_state_variables]

        # forward input through static network
        with torch.no_grad():
            for step in range(steps):
                output = model(inp_tensor[step, :])
                if step % sampling_steps == 0:
                    if verbose:
                        print(f'Progress: {step}/{steps} integration steps finished.')
                    obs.record(output, 0.0, self.rnn_layer.record(rec_vars))

        return obs

    def _compile(self, device: Union[str, None]) -> Sequential:
        in_layer = self._get_layer(self.input_layer)
        out_layer = self._get_layer(self.output_layer)
        rnn_layer = self._get_layer(self.rnn_layer)
        layers = in_layer + rnn_layer + out_layer
        model = Sequential(*layers)
        if device is not None:
            model.to(device)
        return model

    def _relabel_var(self, var: str):
        try:
            return self._var_map[var]
        except KeyError:
            return var

    @staticmethod
    def _get_optimizer(optimizer: str, lr: float, model_params: Iterator, optimizer_kwargs: dict = None
                       ) -> torch.optim.Optimizer:

        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        if optimizer == 'sgd':
            opt = torch.optim.SGD
        elif optimizer == 'adam':
            opt = torch.optim.Adam
        elif optimizer == 'adagrad':
            opt = torch.optim.Adagrad
        elif optimizer == 'lbfgs':
            opt = torch.optim.LBFGS
        elif optimizer == 'rmsprop':
            opt = torch.optim.RMSprop
        else:
            raise ValueError('Invalid optimizer choice. Please see the documentation of the `Network.run()` '
                             'method for valid options.')
        return opt(model_params, lr=lr, **optimizer_kwargs)

    @staticmethod
    def _get_loss_function(loss: str, loss_kwargs: dict = None) -> Callable:

        if loss_kwargs is None:
            loss_kwargs = {}

        if loss == 'mse':
            from torch.nn import MSELoss
            l = MSELoss
        elif loss == 'l1':
            from torch.nn import L1Loss
            l = L1Loss
        elif loss == 'nll':
            from torch.nn import NLLLoss
            l = NLLLoss
        elif loss == 'ce':
            from torch.nn import CrossEntropyLoss
            l = CrossEntropyLoss
        elif loss == 'kld':
            from torch.nn import KLDivLoss
            l = KLDivLoss
        elif loss == 'hinge':
            from torch.nn import HingeEmbeddingLoss
            l = HingeEmbeddingLoss
        else:
            raise ValueError('Invalid loss function choice. Please see the documentation of the `Network.run()` '
                             'method for valid options.')
        return l(**loss_kwargs)

    @staticmethod
    def _get_layer(layer) -> tuple:
        if layer is None:
            return tuple()
        if hasattr(layer, 'layer'):
            return tuple([layer.layer])
        return tuple([layer])
