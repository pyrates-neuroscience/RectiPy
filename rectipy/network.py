import torch
from torch.nn import Sequential
from typing import Union, Iterator, Callable, Tuple, Optional
from .rnn_layer import RNNLayer, SRNNLayer
from .ffwd_layer import LayerStack, RLSLayer, Linear
from .utility import retrieve_from_dict, add_op_name
from .observer import Observer
from pyrates import NodeTemplate, CircuitTemplate
import numpy as np
from time import perf_counter
from multipledispatch import dispatch


class Network:
    """Main user interface for initializing, training, testing, and running networks consisting of rnn, input, and
    output layers.
    """

    def __init__(self, n: int, rnn_layer: Union[RNNLayer, SRNNLayer], var_map: dict = None, device: str = "cpu"):
        """Instantiates network with a single RNN layer.

        Parameters
        ----------
        n
            Number of neurons in the RNN layer.
        rnn_layer
            `RNNLayer` instance.
        var_map
            Optional dictionary, where keys are variable names as supplied by the user, and values are variable names
            that contain the operator name as well.
        device
            Device on which to deploy the `Network` instance.

        """

        self.n = n
        self.rnn_layer = rnn_layer
        self.input_layer = None
        self.output_layer = None
        self.feedback_layer = None
        self._var_map = var_map if var_map else {}
        self._model = None
        self.device = device
        
    def __getitem__(self, item: Union[int, str]):
        try:
            return self.rnn_layer[item]
        except KeyError:
            try:
                return self.rnn_layer[self._var_map[item]]
            except KeyError:
                if self._model is None:
                    self.compile()
                return self._model[item]

    def __len__(self):
        try:
            return len(self._model)
        except TypeError:
            self.compile()
            return len(self._model)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @classmethod
    def from_yaml(cls, node: Union[str, NodeTemplate], weights: np.ndarray, source_var: str, target_var: str,
                  input_var: str, output_var: str, spike_var: str = None, spike_def: str = None, op: str = None,
                  train_params: list = None, device: str = "cpu", **kwargs):
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
        input_var
            Name of the parameter in the node equations that input from input layers should be projected to.
        output_var
            Name of the variable in the node equations that should be used as output of the RNN layer.
        spike_var
            Name of the parameter in the node equations that recurrent input from the RNN layer should be projected to.
        spike_def
            Name of the variable in the node equations that should be used to determine spikes in the network.
        op
            Name of the operator in which all the above variables can be found. If not provided, it is assumed that
            the operator name is provided together with the variable names, e.g. `source_var = <op>/<var>`.
        train_params
            Names of all RNN parameters that should be made available for optimization.
        device
            Device on which to deploy the `Network` instance.
        kwargs
            Additional keyword arguments provided to the `RNNLayer` (or `SRNNLayer` in case of spiking neurons).

        Returns
        -------
        Network
            Instance of `Network`.
        """
        # TODO: Allow to create networks without any recurrent weights (recurrence can also come from feedback)

        # add operator key to variable names
        var_dict = {'svar': source_var, 'tvar': target_var, 'in_ext': input_var, 'in_net': spike_var,
                    'out': output_var, 'spike': spike_def}
        new_vars = {}
        if op is not None:
            for key, var in var_dict.copy().items():
                var_dict[key] = add_op_name(op, var, new_vars)
            if train_params:
                train_params = [add_op_name(op, p, new_vars) for p in train_params]
            if "node_vars" in kwargs:
                for key in kwargs["node_vars"].copy():
                    if "/" not in key:
                        val = kwargs["node_vars"].pop(key)
                        kwargs["node_vars"][f"all/{op}/{key}"] = val

        # initialize rnn layer
        if spike_var is None and spike_def is None:
            rnn_layer = RNNLayer.from_yaml(node, weights, var_dict['svar'], var_dict['tvar'], var_dict['in_ext'],
                                           var_dict['out'], train_params=train_params, device=device, **kwargs)
        elif spike_var is None or spike_def is None:
            raise ValueError('To define a reservoir with a spiking neural network layer, please provide both the '
                             'name of the variable that spikes should be stored in (`spike_var`) as well as the '
                             'name of the variable that is used to define spikes (`spike_def`).')
        else:
            rnn_layer = SRNNLayer.from_yaml(node, weights, var_dict['svar'], var_dict['tvar'], var_dict['in_ext'],
                                            var_dict['out'], spike_def=var_dict['spike'], spike_var=var_dict['in_net'],
                                            train_params=train_params, device=device, **kwargs)

        # remember operator mapping for each RNN layer parameter and state variable
        for p in rnn_layer.parameter_names:
            add_op_name(op, p, new_vars)
        for v in rnn_layer.variable_names:
            add_op_name(op, v, new_vars)

        # initialize model
        return cls(weights.shape[0], rnn_layer, var_map=new_vars, device=device)

    @classmethod
    def from_template(cls, template: CircuitTemplate, input_var: str, output_var: str, spike_var: str = None,
                      spike_def: str = None, op: str = None, train_params: list = None, device: str = "cpu", **kwargs):
        """Creates a `Network` instance from a YAML template that defines a single RNN node and additional information
        about which nodes in the network should be connected to each other.

        Parameters
        ----------
        template
            Instance of a `pyrates.CircuitTemplate`. Will not be altered any further.
        input_var
            Name of the parameter in the node equations that input from input layers should be projected to.
        output_var
            Name of the variable in the node equations that should be used as output of the RNN layer.
        spike_var
            Name of the parameter in the node equations that recurrent input from the RNN layer should be projected to.
        spike_def
            Name of the variable in the node equations that should be used to determine spikes in the network.
        op
            Name of the operator in which all the above variables can be found. If not provided, it is assumed that
            the operator name is provided together with the variable names, e.g. `source_var = <op>/<var>`.
        train_params
            Names of all RNN parameters that should be made available for optimization.
        device
            Device on which to deploy the `Network` instance.
        kwargs
            Additional keyword arguments provided to the `RNNLayer` (or `SRNNLayer` in case of spiking neurons).

        Returns
        -------
        Network
            Instance of `Network`.
        """

        # add operator key to variable names
        var_dict = {'in_ext': input_var, 'in_net': spike_var, 'out': output_var, 'spike': spike_def}
        new_vars = {}
        if op is not None:
            for key, var in var_dict.copy().items():
                var_dict[key] = add_op_name(op, var, new_vars)
            if train_params:
                train_params = [add_op_name(op, p, new_vars) for p in train_params]

        # initialize rnn layer
        if spike_var is None and spike_def is None:
            rnn_layer = RNNLayer.from_template(template, var_dict['in_ext'], var_dict['out'], train_params=train_params,
                                               **kwargs)
        elif spike_var is None or spike_def is None:
            raise ValueError('To define a reservoir with a spiking neural network layer, please provide both the '
                             'name of the variable that spikes should be stored in (`spike_var`) as well as the '
                             'name of the variable that is used to define spikes (`spike_def`).')
        else:
            rnn_layer = SRNNLayer.from_template(template, var_dict['in_ext'], var_dict['out'],
                                                spike_def=var_dict['spike'], spike_var=var_dict['in_net'],
                                                train_params=train_params, **kwargs)

        # remember operator mapping for each RNN layer parameter and state variable
        for p in rnn_layer.parameter_names:
            add_op_name(op, p, new_vars)
        for v in rnn_layer.variable_names:
            add_op_name(op, v, new_vars)

        # initialize model
        return cls(len(template.nodes), rnn_layer, var_map=new_vars, device=device)

    @property
    def model(self) -> Sequential:
        """After `Network.compile` was called, this property yields the `torch.nn.Sequential` instance that contains all
        the network layers.
        """
        return self._model

    def add_input_layer(self, m: int, weights: np.ndarray = None, train: Optional[str] = None,
                        dtype: torch.dtype = torch.float64, **kwargs) -> Linear:
        """Add an input layer to the network. Networks can have either 1 or 0 input layers.

        Parameters
        ----------
        m
            Number of input dimensions.
        weights
            `n x m` weight matrix that realizes the linear projection of the inputs in each input dimension to the
            units in the RNN layer.
        train
            Can be used to make the output layer trainable. The following options are available:
            - `None` for a static output layer
            - 'gd' for training of the readout weights via standard pytorch gradient descent
        dtype
            Data type of the input weights.
        kwargs
            Additional keyword arguments to be passed to the input layer class initialization method.

        Returns
        -------
        Linear
            Instance of the input layer.
        """

        # initialize input layer
        kwargs.update({"n_in": m, "n_out": self.n, "weights": weights, "dtype": dtype})
        if train is None:
            input_layer = Linear(n_in=m, n_out=self.n, weights=weights, dtype=dtype, detach=True)
        elif train == "gd":
            input_layer = Linear(n_in=m, n_out=self.n, weights=weights, dtype=dtype, detach=False)
        else:
            raise ValueError("Invalid option for keyword argument `train`. Please see the docstring of "
                             "`Network.add_input_layer` for valid options.")

        # add layer to model
        self.input_layer = input_layer.to(self.device)

        # return layer
        return self.input_layer

    def add_output_layer(self, k: int, weights: np.ndarray = None, train: Optional[str] = None,
                         activation_function: str = None, dtype: torch.dtype = torch.float64, **kwargs
                         ) -> Union[Linear, LayerStack]:
        """Add an output layer to the network. Networks can have either 1 or 0 output layers.

        Parameters
        ----------
        k
            Number of output dimensions.
        weights
            `k x n` weight matrix that realizes the linear projection of the output of the RNN layer units to the output
            layer units.
        train
            Can be used to make the output layer trainable. The following options are available:
            - `None` for a static output layer
            - 'gd' for training of the readout weights via standard pytorch gradient descent
            - 'rls' for recursive least squares training of the readout weights
        activation_function
            Optional activation function applied to the output of the output layer. Valid options are:
            - 'tanh' for `torch.nn.Tanh()`
            - 'sigmoid' for `torch.nn.Sigmoid()`
            - 'softmax' for `torch.nn.Softmax(dim=0)`
            - `softmin` for `torch.nn.Softmin(dim=0)`
            - `None` (default) for `torch.nn.Identity()`
        dtype
            Data type of the input weights.
        kwargs
            Additional keyword arguments to be passed to the output layer class initialization method.

        Returns
        -------
        Linear
            Instance of the output layer.
        """

        # initialize output layer
        kwargs.update({"n_in": self.n, "n_out": k, "weights": weights, "dtype": dtype})
        if train is None:
            output_layer = Linear(**kwargs, detach=True)
        elif train == "gd":
            output_layer = Linear(**kwargs, detach=False)
        elif train == "rls":
            output_layer = RLSLayer(**kwargs)
        else:
            raise ValueError("Invalid option for keyword argument `train`. Please see the docstring of "
                             "`Network.add_output_layer` for valid options.")

        # add activation function to output layer
        if activation_function:
            output_layer = LayerStack(output_layer, activation_function)

        # add layer to model
        self.output_layer = output_layer.to(self.device)

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

    def train(self, inputs: np.ndarray, targets: np.ndarray, method: str = "gradient_descent",
              sampling_steps: int = 100, verbose: bool = True, **kwargs) -> Observer:
        """High-level training method for model parameter optimization. Allows to choose between the specific
        optimization methods.

        Parameters
        ----------
        inputs
            `T x m` array of inputs fed to the model, where`T` is the number of training steps and `m` is the number of
            input dimensions of the network.
        targets
            `T x k` array of targets, where `T` is the number of training steps and `k` is the number of outputs of the
            network.
        method
            Name of the optimization method. Possible choices are:
            - 'gradient_descent' for gradient-descent-based optimization via `torch.autograd`
            - 'rls' for recursive least-squares
        sampling_steps
            Number of training steps at which to record observables.
        verbose
            If true, the training progress will be displayed.
        kwargs
            Additional keyword arguments passed to the chosen training method.

        Returns
        -------
        Observer
            Instance of the `observer`.
        """

        if method == "gradient_descent":
            return self.train_gd(inputs, targets, sampling_steps=sampling_steps, verbose=verbose, **kwargs)
        if method == "rls":
            return self.train_rls(inputs, targets, sampling_steps=sampling_steps, verbose=verbose, **kwargs)
        if method == "force":
            if "feedback_weights" not in kwargs:
                n_out = inputs.shape[1] if self.input_layer is None else self.input_layer.weights.shape[0]
                n_in = targets.shape[1]
                kwargs["feedback_weights"] = torch.randn(n_out, n_in, device=self.device, dtype=targets.dtype)
            return self.train_rls(inputs, targets, sampling_steps=sampling_steps, verbose=verbose, **kwargs)
        else:
            raise ValueError("Invalid training method. Please see the docstring of `Network.train` for valid choices "
                             "of the keyword argument 'method'.")

    def train_gd(self, inputs: np.ndarray, targets: np.ndarray, optimizer: str = 'sgd', optimizer_kwargs: dict = None,
                 loss: str = 'mse', loss_kwargs: dict = None, lr: float = 1e-3, sampling_steps: int = 100,
                 update_steps: int = 1, verbose: bool = True, **kwargs) -> Observer:
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
            - 'adamw' for torch.optim.AdamW
            - 'adagrad' for `torch.optim.Adagrad`
            - 'adadelta' for `torch.optim.Adadelta`
            - 'rmsprop' for `torch.optim.RMSprop`
            - 'rprop' for `torch.optim.Rprop`
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
        sampling_steps
            Number of training steps at which to record observables.
        update_steps
            Number of training steps after which to perform an update of the trainable parameters based on the
            accumulated gradients.
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
        inp_tensor = torch.tensor(inputs, device=self.device)
        target_tensor = torch.tensor(targets, device=self.device)
        if inp_tensor.shape[0] != target_tensor.shape[0]:
            raise ValueError('Wrong dimensions of input and target output. Please make sure that `inputs` and '
                             '`targets` agree in the first dimension.')

        # set up model
        model = self.compile() if self._model is None else self._model

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

        t0 = perf_counter()
        if len(inp_tensor.shape) > 2:
            self._train_gd_epochs(inp_tensor, target_tensor, model, loss, optimizer, obs, rec_vars, error_kwargs,
                                  step_kwargs, sampling_steps=sampling_steps, optim_steps=update_steps,
                                  verbose=verbose)
        else:
            self._train_gd(inp_tensor, target_tensor, model, loss, optimizer, obs, rec_vars, error_kwargs, step_kwargs,
                           sampling_steps=sampling_steps, optim_steps=update_steps, verbose=verbose)
        t1 = perf_counter()
        print(f'Finished optimization after {t1-t0} s.')
        return obs

    def train_rls(self, inputs: np.ndarray, targets: np.ndarray,  feedback_weights: np.ndarray = None,
                  update_steps: int = 1, sampling_steps: int = 100, verbose: bool = True, **kwargs) -> Observer:
        r"""Finds model parameters $w$ such that $||Xw - y||_2$ is minimized, where $X$ contains the neural activity and
        $y$ contains the targets.

        Parameters
        ----------
        inputs
            `T x m` array of inputs fed to the model, where`T` is the number of training steps and `m` is the number of
            input dimensions of the network.
        targets
            `T x k` array of targets, where `T` is the number of training steps and `k` is the number of outputs of the
            network.
        feedback_weights
            `m x k` array of synaptic weights. If provided, a feedback connections is established with these weights,
            that projects the network output back to the RNN layer.
        update_steps
            Each `update_steps` an update of the trainable parameters will be performed.
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

        # test correct dimensionality of inputs
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError('Wrong dimensions of input and target output. Please make sure that `inputs` and '
                             '`targets` agree in the first dimension.')

        # transform inputs into tensors
        inp_tensor = torch.tensor(inputs, device=self.device)
        target_tensor = torch.tensor(targets, device=self.device)

        # set up model
        if self.output_layer is None:
            self.add_output_layer(k=targets.shape[1], train="rls", **kwargs)
        if self._model is None:
            self.compile()

        # initialize observer
        obs_kwargs = retrieve_from_dict(['record_output', 'record_loss', 'record_vars'], kwargs)
        obs = Observer(dt=self.rnn_layer.dt, **obs_kwargs)
        rec_vars = [self._relabel_var(v) for v in obs.recorded_rnn_variables]

        # optimization
        ##############

        t0 = perf_counter()
        if feedback_weights is None:
            obs = self._train_nofb(inp_tensor, target_tensor, obs, rec_vars, update_steps=update_steps,
                                   sampling_steps=sampling_steps, verbose=verbose, **kwargs)
        else:
            W_fb = torch.tensor(feedback_weights, device=self.device)
            obs = self._train_fb(inp_tensor, target_tensor, W_fb, obs, rec_vars, update_steps=update_steps,
                                 sampling_steps=sampling_steps, verbose=verbose, **kwargs)
        t1 = perf_counter()
        print(f'Finished optimization after {t1 - t0} s.')
        return obs

    def train_rl(self, inputs: np.ndarray, targets: np.ndarray,  feedback_weights: np.ndarray = None,
                 epsilon: float = 0.99, delta: float = 0.9, update_steps: int = 1, sampling_steps: int = 100,
                 verbose: bool = True, **kwargs) -> Observer:
        r"""Reinforcement learning algorithm that implements slow adjustment of the feedback weights to the RNN layer
        based on a running average of the residuals.

        Parameters
        ----------
        inputs
            `T x m` array of inputs fed to the model, where`T` is the number of training steps and `m` is the number of
            input dimensions of the network.
        targets
            `T x k` array of targets, where `T` is the number of training steps and `k` is the number of outputs of the
            network.
        feedback_weights
            `m x k` array of synaptic weights. If provided, a feedback connections is established with these weights,
            that projects the network output back to the RNN layer.
        epsilon
            Scalar in (0, 1] that controls how quickly the loss used for reinforcement learning can change.
        delta
            Scalar in (0, 1] that controls how quickly the feedback weights can change.
        update_steps
            Each `update_steps` an update of the trainable parameters will be performed.
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

        # test correct dimensionality of inputs
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError('Wrong dimensions of input and target output. Please make sure that `inputs` and '
                             '`targets` agree in the first dimension.')

        # transform inputs into tensors
        inp_tensor = torch.tensor(inputs, device=self.device)
        target_tensor = torch.tensor(targets, device=self.device)

        # set up model
        if self.output_layer is None:
            self.add_output_layer(k=targets.shape[1], train="rls", **kwargs)
        if self._model is None:
            self.compile()

        # initialize observer
        obs_kwargs = retrieve_from_dict(['record_output', 'record_loss', 'record_vars'], kwargs)
        obs = Observer(dt=self.rnn_layer.dt, **obs_kwargs)
        rec_vars = [self._relabel_var(v) for v in obs.recorded_rnn_variables]

        # optimization
        ##############

        t0 = perf_counter()
        if feedback_weights is None:
            feedback_weights = np.random.randn(self.rnn_layer.n, self.rnn_layer.n)
        W_fb = torch.tensor(feedback_weights, device=self.device)
        obs = self._train_rl(inp_tensor, target_tensor, W_fb, epsilon, delta, obs, rec_vars, update_steps=update_steps,
                             sampling_steps=sampling_steps, verbose=verbose, **kwargs)
        t1 = perf_counter()
        print(f'Finished optimization after {t1 - t0} s.')
        return obs

    def test(self, inputs: np.ndarray, targets: np.ndarray, feedback_weights: np.ndarray = None, loss: str = 'mse',
             loss_kwargs: dict = None, sampling_steps: int = 100, verbose: bool = True, **kwargs) -> tuple:
        """Test the model performance on a set of inputs and target outputs, with frozen model parameters.

        Parameters
        ----------
        inputs
            `T x m` array of inputs fed to the model, where`T` is the number of testing steps and `m` is the number of
            input dimensions of the network.
        targets
            `T x k` array of targets, where `T` is the number of testing steps and `k` is the number of outputs of the
            network.
        feedback_weights
            `m x k` array of synaptic weights. If provided, a feedback connections is established with these weights,
            that projects the network output back to the RNN layer.
        loss
            Name of the loss function that should be used to calculate the loss on the test data. See `Network.train`
            for available options.
        loss_kwargs
            Additional keyword arguments provided to the initialization of the loss.
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
        inp_tensor = torch.tensor(inputs, device=self.device)
        target_tensor = torch.tensor(targets, device=self.device)

        # set up model
        if self._model is None:
            self.compile()

        # initialize loss function
        loss = self._get_loss_function(loss, loss_kwargs=loss_kwargs)

        # initialize observer
        obs_kwargs = retrieve_from_dict(['record_output', 'record_loss', 'record_vars'], kwargs)
        obs = Observer(dt=self.rnn_layer.dt, **obs_kwargs)
        rec_vars = [self._relabel_var(v) for v in obs.recorded_rnn_variables]

        # call testing method
        if feedback_weights is None:
            obs, loss_total = self._test_nofb(inp_tensor, target_tensor, loss, obs, rec_vars, sampling_steps, verbose)
        else:
            W_fb = torch.tensor(feedback_weights, device=self.device)
            obs, loss_total = self._test_fb(inp_tensor, target_tensor, W_fb, loss, obs, rec_vars, sampling_steps,
                                            verbose)

        return obs, loss_total

    def run(self, inputs: np.ndarray, sampling_steps: int = 1, verbose: bool = True, **kwargs
            ) -> Observer:
        """Perform numerical integration of the input-driven network equations.

        Parameters
        ----------
        inputs
            `T x m` array of inputs fed to the model, where`T` is the number of integration steps and `m` is the number
             of input dimensions of the network.
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
        inp_tensor = torch.tensor(inputs, device=self.device)

        # initialize model from layers
        model = self.compile() if self._model is None else self._model

        # initialize observer
        obs = Observer(dt=self.rnn_layer.dt, record_loss=kwargs.pop("record_loss", False), **kwargs)
        rec_vars = [self._relabel_var(v) for v in obs.recorded_rnn_variables]

        # forward input through static network
        with torch.no_grad():
            for step in range(steps):
                output = model(inp_tensor[step, :])
                if step % sampling_steps == 0:
                    if verbose:
                        print(f'Progress: {step}/{steps} integration steps finished.')
                    obs.record(step, output, 0.0, [self[v] for v in rec_vars])

        return obs

    def compile(self) -> Sequential:
        """Connects the `InputLayer`, `RNNLayer` and `OutputLayer` of this model via a `torch.nn.Sequential` instance.
        Input and output layers are optional.

        Parameters
        ----------

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
        self._model = model.to(self.device)
        if not list(self.rnn_layer.parameters()):
            self.rnn_layer.detach()
        else:
            for p in self.parameters():
                p.requires_grad = True
        return self._model

    @dispatch(object)
    def forward(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Forward method as implemented for any `torch.Module`.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        return self._model(x)

    @dispatch(object, object)
    def forward(self, x: Union[torch.Tensor, np.ndarray], y: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Forward method for a network model that receives additional feedback from the last layer to the rnn layer.

        Parameters
        ----------
        x
            Input tensor.
        y
            Feedback tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        if len(self._model) < 2:
            return self._model(x + y)
        pos = 0 if isinstance(self._model[0], RNNLayer) else 1
        for layer in self._model[:pos]:
            x = layer(x)
        x = self._model[pos](x + y)
        try:
            return self._model[pos+1](x)
        except IndexError:
            return x

    def parameters(self, recurse: bool = True) -> Iterator:
        """Yields the trainable parameters of the network model.

        Parameters
        ----------
        recurse
            If true, yields parameters of all submodules.

        Yields
        ------
        Iterator
            Trainable model parameters.
        """
        if self._model is None:
            self.compile()
        for layer in self._model:
            for p in layer.parameters(recurse):
                yield p

    def _train_gd(self, inp: torch.Tensor, target: torch.Tensor, model: Sequential, loss: Callable,
                  optimizer: torch.optim.Optimizer, obs: Observer, rec_vars: list, error_kwargs: dict, step_kwargs: dict,
                  sampling_steps: int = 100, optim_steps: int = 1, verbose: bool = False) -> Observer:

        steps = inp.shape[0]
        for step in range(steps):

            # forward pass
            prediction = model(inp[step, :])

            # loss calculation
            error = loss(prediction, target[step, :])

            # error backpropagation
            error.backward(**error_kwargs)
            if step % optim_steps == 0:
                optimizer.step(**step_kwargs)
                optimizer.zero_grad()

            # results storage
            if step % sampling_steps == 0:
                if verbose:
                    print(f'Progress: {step}/{steps} training steps finished.')
                obs.record(step, prediction, error.item(), [self[v] for v in rec_vars])

        return obs

    def _train_gd_epochs(self, inp: torch.Tensor, target: torch.Tensor, model: Sequential, loss: Callable,
                         optimizer: torch.optim.Optimizer, obs: Observer, rec_vars: list, error_kwargs: dict,
                         step_kwargs: dict, sampling_steps: int = 100, optim_steps: int = 1, verbose: bool = False
                         ) -> Observer:

        if inp.shape[1] != target.shape[1]:
            raise ValueError('Wrong dimensions of input and target output. Please make sure that `inputs` and '
                             '`targets` agree in the first dimension (epochs) and second dimension (steps per epoch).')

        epochs = inp.shape[0]
        steps = inp.shape[1]
        for epoch in range(epochs):

            # go through steps of the epoch
            epoch_loss = 0
            for step in range(steps):

                # forward pass
                prediction = model(inp[epoch, step, :])

                # loss calculation
                error = loss(prediction, target[epoch, step, :])
                epoch_loss += error.item()

                # error backpropagation
                error.backward(**error_kwargs)
                if step % optim_steps == 0:
                    optimizer.step(**step_kwargs)
                    optimizer.zero_grad()

                # results storage
                if step % sampling_steps == 0:
                    obs.record(step, prediction, error.item(), [self[v] for v in rec_vars])

            # display progress
            if verbose:
                print(f'Progress: {epoch+1}/{epochs} training epochs finished.')
                print(f'Epoch loss: {epoch_loss}.')
                print('')

        return obs

    def _train_nofb(self, inp: torch.Tensor, targets: torch.Tensor, obs: Observer, rec_vars: list, update_steps: int = 1,
                    sampling_steps: int = 100, verbose: bool = False, tol: float = 1e-3, loss_beta: float = 0.9
                    ) -> Observer:

        out_layer = self._model.pop(-1)
        steps = inp.shape[0]
        loss = 1.0
        with torch.no_grad():
            for step in range(steps):

                if step % update_steps == 0:

                    # perform weight update
                    x = self.forward(inp[step, :])
                    y_hat = out_layer.forward(x)
                    y = targets[step, :]
                    out_layer.update(x, y_hat, y)

                else:

                    # perform forward pass
                    y_hat = self.forward(inp[step, :])

                # results storage
                if step % sampling_steps == 0:
                    if torch.isnan(out_layer.loss):
                        print("ABORTING NETWORK TRAINING: Loss in output layer evaluated to a non-finite number.")
                        break
                    loss_tmp = out_layer.loss
                    obs.record(step, y_hat, loss_tmp, [self[v] for v in rec_vars])
                    loss = loss_beta * loss + (1.0 - loss_beta) * loss_tmp
                    if verbose:
                        print(f'Progress: {step}/{steps} training steps finished.')
                        print(f'Current loss: {loss}.')
                        print('')

                    # break condition
                if loss < tol:
                    break

        # add output layer to model again
        self._model.append(out_layer)

        return obs

    def _train_fb(self, inp: torch.Tensor, targets: torch.Tensor, W_fb: torch.Tensor, obs: Observer, rec_vars: list,
                  update_steps: int = 1, sampling_steps: int = 100, verbose: bool = False, tol: float = 1e-3,
                  loss_beta: float = 0.9) -> Observer:

        out_layer = self._model.pop(-1)
        steps = inp.shape[0]

        # get initial step done prior to the loop
        x = self.forward(inp[0, :], W_fb @ targets[0, :])
        y_hat = out_layer.forward(x)
        obs.record(0, y_hat, out_layer.loss, [self[v] for v in rec_vars])
        loss = 1.0
        with torch.no_grad():
            for step in range(steps):

                if step % update_steps == 0:

                    # perform weight update
                    x = self.forward(inp[step, :], W_fb @ y_hat)
                    y_hat = out_layer.forward(x)
                    y = targets[step, :]
                    out_layer.update(x, y_hat, y)

                else:

                    # perform forward pass
                    x = self.forward(inp[step, :], W_fb @ y_hat)
                    y_hat = out_layer.forward(x)

                # results storage
                if step % sampling_steps == 0:
                    if torch.isnan(out_layer.loss):
                        print("ABORTING NETWORK TRAINING: Loss in output layer evaluated to a non-finite number.")
                        break
                    loss_tmp = out_layer.loss
                    obs.record(step, y_hat, loss_tmp, [self[v] for v in rec_vars])
                    loss = loss_beta * loss + (1.0 - loss_beta) * loss_tmp
                    if verbose:
                        print(f'Progress: {step}/{steps} training steps finished.')
                        print(f'Current loss: {loss}.')
                        print('')

                # break condition
                if loss < tol:
                    break

        # add output layer to model again
        self._model.append(out_layer)

        return obs

    def _train_rl(self, inp: torch.Tensor, targets: torch.Tensor, W_fb: torch.Tensor, epsilon: float, delta: float,
                  obs: Observer, rec_vars: list, update_steps: int = 1, sampling_steps: int = 100,
                  verbose: bool = False, tol: float = 1e-3, loss_beta: float = 0.9) -> Observer:

        out_layer = self._model.pop(-1)
        steps = inp.shape[0]

        # extract properties of feedback weights
        sr_0 = torch.max(torch.abs(torch.linalg.eigvals(W_fb)))
        mask = torch.nonzero(W_fb)
        fb_size = W_fb.shape

        # get initial step done prior to the loop
        x = self.forward(inp[0, :])
        y_hat = out_layer.forward(x)
        obs.record(0, y_hat, out_layer.loss, [self[v] for v in rec_vars])
        loss = 1.0
        loss_rl = loss
        with torch.no_grad():
            for step in range(steps):

                if step % update_steps == 0:

                    # perform output layer weight update
                    x = self.forward(inp[step, :], W_fb @ x)
                    y_hat = out_layer.forward(x)
                    y = targets[step, :]
                    out_layer.update(x, y_hat, y)

                else:

                    # perform forward pass
                    x = self.forward(inp[step, :], W_fb @ x)
                    y_hat = out_layer.forward(x)

                # perform feedback weight update
                loss_tmp = out_layer.loss
                loss_rl = 1.0/(1.0 + torch.exp(epsilon*loss_rl + (1-epsilon)*loss_tmp))
                W_new = loss_rl*torch.randn(fb_size, device=self.device) + (1-loss_rl)*out_layer.P/loss
                W_new[mask is False] = 0.0
                W_fb = delta*W_fb + (1-delta)*W_new

                # results storage
                if step % sampling_steps == 0:
                    if torch.isnan(out_layer.loss):
                        print("ABORTING NETWORK TRAINING: Loss in output layer evaluated to a non-finite number.")
                        break
                    obs.record(step, y_hat, loss_tmp, [self[v] for v in rec_vars])
                    loss = loss_beta * loss + (1.0 - loss_beta) * loss_tmp
                    if verbose:
                        print(f'Progress: {step}/{steps} training steps finished.')
                        print(f'Current loss: {loss}.')
                        print('')

                # break condition
                if loss < tol:
                    break

        # add output layer to model again
        self._model.append(out_layer)

        # add feedback matrix to observer
        obs.save("feedback_weights", W_fb)

        return obs

    def _test_nofb(self, inp: torch.Tensor, targets: torch.Tensor, loss: Callable, obs: Observer, rec_vars: list,
                   sampling_steps: int = 1, verbose: bool = True) -> Tuple[Observer, float]:

        loss_total = 0.0
        steps = inp.shape[0]
        with torch.no_grad():
            for step in range(steps):

                # forward pass
                prediction = self.forward(inp[step, :])

                # loss calculation
                error = loss(prediction, targets[step, :])
                loss_total += error.item()

                # results storage
                if step % sampling_steps == 0:
                    if verbose:
                        print(f'Progress: {step}/{steps} test steps finished.')
                    obs.record(step, prediction, error.item(), [self[v] for v in rec_vars])

        return obs, loss_total

    def _test_fb(self, inp: torch.Tensor, targets: torch.Tensor, W_fb: torch.Tensor, loss: Callable, obs: Observer,
                 rec_vars: list, sampling_steps: int = 1, verbose: bool = True) -> Tuple[Observer, float]:

        loss_total = 0.0
        steps = inp.shape[0]
        prediction = self.forward(inp[0, :], W_fb @ targets[0, :])
        with torch.no_grad():
            for step in range(steps):

                # forward pass
                prediction = self.forward(inp[step, :], W_fb @ prediction)

                # loss calculation
                error = loss(prediction, targets[step, :])
                loss_total += error.item()

                # results storage
                if step % sampling_steps == 0:
                    if verbose:
                        print(f'Progress: {step}/{steps} test steps finished.')
                    obs.record(step, prediction, error.item(), [self[v] for v in rec_vars])

        return obs, loss_total

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
        elif optimizer == 'adamw':
            opt = torch.optim.AdamW
        elif optimizer == 'adagrad':
            opt = torch.optim.Adagrad
        elif optimizer == 'adadelta':
            opt = torch.optim.Adadelta
        elif optimizer == 'rmsprop':
            opt = torch.optim.RMSprop
        elif optimizer == 'rprop':
            opt = torch.optim.Rprop
        else:
            raise ValueError('Invalid optimizer choice. Please see the documentation of the `Network.train()` '
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
            raise ValueError('Invalid loss function choice. Please see the documentation of the `Network.train()` '
                             'method for valid options.')
        return l(**loss_kwargs)

    @staticmethod
    def _get_layer(layer) -> tuple:
        if layer is None:
            return tuple()
        return tuple([layer])
