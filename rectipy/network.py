import torch
from torch.nn import Sequential
from typing import Union, Iterator, Callable, Tuple
from .rnn_layer import RNNLayer, SRNNLayer
from .input_layer import InputLayer
from .output_layer import OutputLayer
from .utility import retrieve_from_dict, add_op_name
from .observer import Observer
from pyrates import NodeTemplate
import numpy as np
from time import perf_counter


class Network:
    """Main user interface for initializing, training, testing, and running networks consisting of rnn, input, and
    output layers.
    """

    def __init__(self, rnn_layer: Union[RNNLayer, SRNNLayer], var_map: dict = None):
        """Instantiates network with a single RNN layer.

        Parameters
        ----------
        rnn_layer
            `RNNLayer` instance.
        var_map
            Optional dictionary, where keys are variable names as supplied by the user, and values are variable names
            that contain the operator name as well.
        """

        self.rnn_layer = rnn_layer
        self.input_layer = None
        self.output_layer = None
        self._var_map = var_map if var_map else {}
        self._model = None

    @classmethod
    def from_yaml(cls, node: Union[str, NodeTemplate], weights: np.ndarray, source_var: str, target_var: str,
                  input_var_ext: str, output_var: str, input_var_net: str = None, spike_var: str = None, op: str = None,
                  train_params: list = None, record_vars: list = None, **kwargs):
        """Creates a `Network` instance from a YAML template that defines a single RNN node and additional information
        about which nodes in the network should be connected to each other.

        Parameters
        ----------
        node
            Path to the YAML template or an instance of a `pyrates.NodeTemplate`.
        weights
            Determines the number of nodes in the network as well as their connectivity. Given an `N x N` weights
            matrix, `N` nodes will be added to the RNN layer, each of which is governed by the equations defined in the
            `NodeTemplate` (see argument `node`). Nodes will be labeled `n0` to `n<N>` and every non-zero entry in the
            matrix will be realized by an edge between the corresponding nodes in the network.
        source_var
            Source variable that will be used for each connection in the network.
        target_var
            Target variable that will be used for each connection in the network.
        input_var_ext
            Name of the parameter in the node equations that input from input layers should be projected to.
        output_var
            Name of the variable in the node equations that should be used as output of the RNN layer.
        input_var_net
            Name of the parameter in the node equations that recurrent input from the RNN layer should be projected to.
        spike_var
            Name of the variable in the node equations that should be used to determine spikes in the network.
        op
            Name of the operator in which all the above variables can be found. If not provided, it is assumed that
            the operator name is provided together with the variable names, e.g. `source_var = <op>/<var>`.
        train_params
            Names of all RNN parameters that should be made available for optimization.
        record_vars
            Names of all variables that should be made available for recordings during training/testing/simulation runs.
        kwargs
            Additional keyword arguments provided to the `RNNLayer` (or `SRNNLayer` in case of spiking neurons).

        Returns
        -------
        Network
            Instance of `Network`.
        """

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

    @property
    def model(self) -> Sequential:
        """After `Network.compile` was called, this property yields the `torch.nn.Sequential` instance that contains all
        the network layers.
        """
        return self._model

    def add_input_layer(self, n: int, m: int, weights: torch.Tensor = None, trainable: bool = False,
                        dtype: torch.dtype = torch.float64) -> InputLayer:
        """Add an input layer to the network. Networks can have either 1 or 0 input layers.

        Parameters
        ----------
        n
            Number of neurons in the RNN layer.
        m
            Number of input dimensions.
        weights
            `n x m` weight matrix that realizes the linear projection of the inputs in each input dimension to the
            neurons in the RNN layer.
        trainable
            If true, the input weights will be made available for optimization.
        dtype
            Data type of the input weights.

        Returns
        -------
        InputLayer
            Instance of the `InputLayer`.
        """

        # initialize input layer
        input_layer = InputLayer(n, m, weights, trainable=trainable, dtype=dtype)

        # add layer to model
        self.input_layer = input_layer

        # return layer
        return self.input_layer

    def add_output_layer(self, n: int, k: int, weights: torch.Tensor = None, trainable: bool = False,
                         activation_function: str = None, dtype: torch.dtype = torch.float64) -> OutputLayer:
        """Add an output layer to the network. Networks can have either 1 or 0 output layers.

        Parameters
        ----------
        n
            Number of neurons in the RNN layer.
        k
            Number of output dimensions.
        weights
            `k x n` weight matrix that realizes the linear projection of the output of the RNN layer units to the output
            layer units.
        trainable
            If true, the output weights will be made available for optimization.
        activation_function
            Optional activation function applied to the output of the output layer.
        dtype
            Data type of the input weights.

        Returns
        -------
        OutputLayer
            Instance of the `OutputLayer`.
        """

        # initialize output layer
        output_layer = OutputLayer(n, k, weights, trainable=trainable, activation_function=activation_function,
                                   dtype=dtype)

        # add layer to model
        self.output_layer = output_layer

        # return layer
        return self.output_layer

    def remove_input_layer(self) -> None:
        """Removes the current input layer.
        """
        self.input_layer = None

    def remove_output_layer(self) -> None:
        """Removes the current output layer.
        """
        self.output_layer = None

    def train(self, inputs: np.ndarray, targets: np.ndarray, optimizer: str = 'sgd', optimizer_kwargs: dict = None,
              loss: str = 'mse', loss_kwargs: dict = None, lr: float = 1e-3, device: str = None,
              sampling_steps: int = 100, verbose: bool = True, **kwargs) -> Observer:
        """Optimize model parameters such that the model output matches the provided targets as close as possible.

        Parameters
        ----------
        inputs
            `T x m` array of inputs fed to the model, where`T` is the number of training steps and `m` is the number of
            input dimensions of the network.
        targets
            `T x k` array of targets, where `T` is the number of training steps and `k` is the number of outputs of the
            network.
        optimizer
            Name of the optimization algorithm to use. Available options are:
            - 'sgd' for `torch.optim.SGD`
            - 'adam' for `torch.optim.Adam`
            - 'adagrad' for `torch.optim.Adagrad`
            - 'lbfgs' for `torch.optim.LBFGS`
            - 'rmsprop' for `torch.optim.RMSprop`
        optimizer_kwargs
            Additional keyword arguments provided to the initialization of the optimizer.
        loss
            Name of the loss function that should be used for optimization. Available options are:
            - 'mse' for `torch.nn.MSELoss`
            - 'l1' for `torch.nn.L1Loss`
            - 'nll' for `torch.nn.NLLLoss`
            - 'ce' for `torch.nn.CrossEntropyLoss`
            - 'kld' for `torch.nn.KLDivLoss`
            - 'hinge' for `torch.nn.HingeEmbeddingLoss`
        loss_kwargs
            Additional keyword arguments provided to the initialization of the loss.
        lr
            Learning rate.
        device
            Device on which to deploy the network model.
        sampling_steps
            Number of training steps at which to record observables.
        verbose
            If true, the training progress will be displayed.
        kwargs
            Additional keyword arguments used for the optimization, loss calculation and observation.

        Returns
        -------
        Observer
            Instance of the `observer`.
        """

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
        model = self.compile(device) if self._model is None else self._model

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
        rec_vars = [self._relabel_var(v) for v in obs.recorded_rnn_variables]

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
        """Test the model performance on a set of inputs and target outputs, with frozen model parameters.

        Parameters
        ----------
        inputs
            `T x m` array of inputs fed to the model, where`T` is the number of testing steps and `m` is the number of
            input dimensions of the network.
        targets
            `T x k` array of targets, where `T` is the number of testing steps and `k` is the number of outputs of the
            network.
        loss
            Name of the loss function that should be used to calculate the loss on the test data. See `Network.train`
            for available options.
        loss_kwargs
            Additional keyword arguments provided to the initialization of the loss.
        device
            Device on which to deploy the network model.
        sampling_steps
            Number of testing steps at which to record observables.
        verbose
            If true, the progress of the test run will be displayed.
        kwargs
            Additional keyword arguments used for the loss calculation and observation.

        Returns
        -------
        Tuple[Observer,float]
            The `Observer` instance and the total loss on the test data.
        """

        # preparations
        ##############

        # transform inputs into tensors
        inp_tensor = torch.tensor(inputs)
        target_tensor = torch.tensor(targets)
        if inp_tensor.shape[0] != target_tensor.shape[0]:
            raise ValueError('Wrong dimensions of input and target output. Please make shure that `inputs` and '
                             '`targets` agree in the first dimension.')

        # set up model
        model = self.compile(device) if self._model is None else self._model

        # initialize loss function
        loss = self._get_loss_function(loss, loss_kwargs=loss_kwargs)

        # initialize observer
        obs_kwargs = retrieve_from_dict(['record_output', 'record_loss', 'record_vars'], **kwargs)
        obs = Observer(dt=self.rnn_layer.dt, **obs_kwargs)
        rec_vars = [self._relabel_var(v) for v in obs.recorded_rnn_variables]

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
        """Perform numerical integration of the input-driven network equations.

        Parameters
        ----------
        inputs
            `T x m` array of inputs fed to the model, where`T` is the number of integration steps and `m` is the number
             of input dimensions of the network.
        device
            Device on which to deploy the network model.
        sampling_steps
            Number of integration steps at which to record observables.
        verbose
            If true, the progress of the integration will be displayed.
        kwargs
            Additional keyword arguments used for the observation.

        Returns
        -------
        Observer
            Instance of the `Observer`.
        """

        # transform input into tensor
        steps = inputs.shape[0]
        inp_tensor = torch.tensor(inputs)

        # initialize model from layers
        model = self.compile(device) if self._model is None else self._model

        # initialize observer
        obs = Observer(dt=self.rnn_layer.dt, record_loss=False, **kwargs)
        rec_vars = [self._relabel_var(v) for v in obs.recorded_rnn_variables]

        # forward input through static network
        with torch.no_grad():
            for step in range(steps):
                output = model(inp_tensor[step, :])
                if step % sampling_steps == 0:
                    if verbose:
                        print(f'Progress: {step}/{steps} integration steps finished.')
                    obs.record(output, 0.0, self.rnn_layer.record(rec_vars))

        return obs

    def compile(self, device: Union[str, None]) -> Sequential:
        """Connects the `InputLayer`, `RNNLayer` and `OutputLayer` of this model via a `torch.nn.Sequential` instance.
        Input and output layers are optional.

        Parameters
        ----------
        device
            Device on which to deploy the `torch.nn.Sequential` instance.

        Returns
        -------
        Sequential
            Instance of the `torch.nn.Sequential` containing the network layers.
        """
        in_layer = self._get_layer(self.input_layer)
        out_layer = self._get_layer(self.output_layer)
        rnn_layer = self._get_layer(self.rnn_layer)
        layers = in_layer + rnn_layer + out_layer
        model = Sequential(*layers)
        if device is not None:
            model.to(device)
        return model

    def _relabel_var(self, var: str) -> str:
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
