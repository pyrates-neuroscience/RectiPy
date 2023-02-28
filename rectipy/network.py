import torch
from torch.nn import Module
from typing import Union, Iterator, Callable, Tuple, Optional
from .nodes import NN, SNN, Function
from .edges import RLS, Linear
from .utility import retrieve_from_dict, add_op_name
from .observer import Observer
from pyrates import NodeTemplate, CircuitTemplate
import numpy as np
from time import perf_counter
from networkx import DiGraph
from multipledispatch import dispatch


class Network(Module):
    """Main user interface for initializing, training, testing, and running networks consisting of rnn, input, and
    output layers.
    """

    def __init__(self, dt: float, device: str = "cpu"):
        """Instantiates network with a single RNN layer.

        Parameters
        ----------
        dt
            Time-step used for all simulations and rnn layers.
        device
            Device on which to deploy the `Network` instance.

        """

        super().__init__()

        self.graph = DiGraph()
        self.device = device
        self.dt = dt
        self._record = {}
        self._layer_map = {}
        self._in_node = None
        self._out_node = None
        self._graph_ffwd = None
        self._graph_fb = None

    def __getitem__(self, item: int):
        return self.graph[item]

    def __len__(self):
        return len(self.graph)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @property
    def n_out(self):
        """Current output dimensionality.
        """
        try:
            return self.graph[self._out_node]["n_out"]
        except AttributeError:
            return 0

    @property
    def n_in(self):
        """Current input dimensionality of the network.
        """
        try:
            return self.graph[self._in_node]["n_in"]
        except AttributeError:
            return 0

    def add_node(self, label: Union[str, int], node_type: str = "function", **kwargs):

        if node_type == "function":
            node = self.add_edge(label, **kwargs)
        elif node_type == "differential_equation":
            node_source = kwargs.pop("node_source", "yaml")
            if node_source == "yaml":
                node = self.add_diffeq_node_from_yaml(label, **kwargs)
            elif node_source == "template":
                node = self.add_diffeq_node_from_template(label, **kwargs)
            else:
                raise ValueError("The node source for differential equation nodes must be either 'yaml' or 'template'.")
        else:
            raise ValueError("Invalid node type.")
        return node

    def add_diffeq_node_from_yaml(self, label: Union[str, int], node: Union[str, NodeTemplate], input_var: str,
                                  output_var: str, weights: np.ndarray = None, source_var: str = None,
                                  target_var: str = None, spike_var: str = None, spike_def: str = None, op: str = None,
                                  train_params: list = None, **kwargs) -> NN:
        """Adds an RNN node to the `Network` instance.

        Parameters
        ----------
        label
            The label of the node in the network graph.
        node
            Path to the YAML template or an instance of a `pyrates.NodeTemplate`.
        input_var
            Name of the parameter in the node equations that input should be projected to.
        output_var
            Name of the variable in the node equations that should be used as output of the RNN node.
        weights
            Determines the number of neurons in the network as well as their connectivity. Given an `N x N` weights
            matrix, `N` neurons will be added to the RNN node, each of which is governed by the equations defined in the
            `NodeTemplate` (see argument `node`). Neurons will be labeled `n0` to `n<N>` and every non-zero entry in the
            matrix will be realized by an edge between the corresponding neurons in the network.
        source_var
            Source variable that will be used for each connection in the network.
        target_var
            Target variable that will be used for each connection in the network.
        spike_var
            Name of the parameter in the node equations that recurrent input from the RNN should be projected to.
        spike_def
            Name of the variable in the node equations that should be used to determine spikes in the network.
        op
            Name of the operator in which all the above variables can be found. If not provided, it is assumed that
            the operator name is provided together with the variable names, e.g. `source_var = <op>/<var>`.
        train_params
            Names of all RNN parameters that should be made available for optimization.
        kwargs
            Additional keyword arguments provided to the `RNNLayer` (or `SRNNLayer` in case of spiking neurons).

        Returns
        -------
        NN
            Instance of the RNN node that was added to the network.
        """

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
            node = NN.from_yaml(node, var_dict['in_ext'], var_dict['out'], weights=weights,
                                source_var=var_dict['svar'], target_var=var_dict['tvar'],
                                train_params=train_params, device=self.device, dt=self.dt, **kwargs)
        elif spike_var is None or spike_def is None:
            raise ValueError('To define a reservoir with a spiking neural network layer, please provide both the '
                             'name of the variable that spikes should be stored in (`spike_var`) as well as the '
                             'name of the variable that is used to define spikes (`spike_def`).')
        else:
            node = SNN.from_yaml(node, var_dict['in_ext'], var_dict['out'], weights=weights,
                                 source_var=var_dict['svar'], target_varw=var_dict['tvar'],
                                 spike_def=var_dict['spike'], spike_var=var_dict['in_net'],
                                 train_params=train_params, device=self.device, dt=self.dt, **kwargs)

        # remember operator mapping for each RNN layer parameter and state variable
        for p in node.parameter_names:
            add_op_name(op, p, new_vars)
        for v in node.variable_names:
            add_op_name(op, v, new_vars)

        # add node to graph
        self.graph.add_node(label, node=node, node_type="diffeq", n_out=node.n_out, n_in=node.n_in)
        return node

    def add_diffeq_node_from_template(self, label: Union[str, int], template: CircuitTemplate, input_var: str,
                                      output_var: str, spike_var: str = None, spike_def: str = None, op: str = None,
                                      train_params: list = None, **kwargs) -> NN:
        """Add RNN node to the `Network` instance from a `pyrates.CircuitTemplate`.

        Parameters
        ----------
        label
            The label of the node in the network graph.
        template
            Instance of a `pyrates.CircuitTemplate`. Will not be altered any further.
        input_var
            Name of the parameter in the node equations that input should be projected to.
        output_var
            Name of the variable in the node equations that should be used as output of the RNN node.
        spike_var
            Name of the parameter in the node equations that recurrent input from the RNN should be projected to.
        spike_def
            Name of the variable in the node equations that should be used to determine spikes in the network.
        op
            Name of the operator in which all the above variables can be found. If not provided, it is assumed that
            the operator name is provided together with the variable names, e.g. `source_var = <op>/<var>`.
        train_params
            Names of all RNN parameters that should be made available for optimization.
        kwargs
            Additional keyword arguments provided to the `RNNLayer` (or `SRNNLayer` in case of spiking neurons).

        Returns
        -------
        NN
            Instance of the RNN layer that was added to the network.
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
            node = NN.from_template(template, var_dict['in_ext'], var_dict['out'], train_params=train_params,
                                    device=self.device, dt=self.dt, **kwargs)
        elif spike_var is None or spike_def is None:
            raise ValueError('To define a reservoir with a spiking neural network layer, please provide both the '
                             'name of the variable that spikes should be stored in (`spike_var`) as well as the '
                             'name of the variable that is used to define spikes (`spike_def`).')
        else:
            node = SNN.from_template(template, var_dict['in_ext'], var_dict['out'],
                                     spike_def=var_dict['spike'], spike_var=var_dict['in_net'],
                                     train_params=train_params, device=self.device, dt=self.dt, **kwargs)

        # remember operator mapping for each RNN layer parameter and state variable
        for p in node.parameter_names:
            add_op_name(op, p, new_vars)
        for v in node.variable_names:
            add_op_name(op, v, new_vars)

        # add node to graph
        self.graph.add_node(label, node=node, node_type="diffeq", n_out=node.n_out, n_in=node.n_in)
        return node

    def add_func_node(self, label: Union[int, str], activation_function: str, **kwargs) -> Function:
        """Add an activation function as a node to the network (no intrinsic dynamics, just an input-output mapping).

        Parameters
        ----------
        label
            The label of the node in the network graph.
        activation_function
            Activation function applied to the output of the last layer. Valid options are:
            - 'tanh' for `torch.nn.Tanh()`
            - 'sigmoid' for `torch.nn.Sigmoid()`
            - 'softmax' for `torch.nn.Softmax(dim=0)`
            - `softmin` for `torch.nn.Softmin(dim=0)`

        Returns
        -------
        ActivationFunc
            The node of the network graph.
        """
        node = Function(activation_function, **kwargs)
        self.graph.add_node(label, node=node, node_type="func", n_out=node.n_out, n_in=node.n_in)
        return node

    def add_edge(self, source: Union[str, int], target: Union[str, int], weights: np.ndarray = None,
                 train: Optional[str] = None, feedback: bool = False, dtype: torch.dtype = torch.float64, **kwargs
                 ) -> Linear:
        """Add a feed-forward layer to the network.

        Parameters
        ----------
        source
            Label of the source node.
        target
            Label of the target node.
        weights
            `k x n` weight matrix that realizes the linear projection of the `n` source outputs to
            the `k` target inputs.
        train
            Can be used to make the edge weights trainable. The following options are available:
            - `None` for a static edge
            - 'gd' for training of the edge weights via standard pytorch gradient descent
            - 'rls' for recursive least squares training of the edge weights
        feedback
            If true, this edge is treated as a feedback edge, meaning that it does not affect the feedforward path that
            connects the network input to its output.
        dtype
            Data type of the edge weights.
        kwargs
            Additional keyword arguments to be passed to the edge class initialization method.

        Returns
        -------
        Linear
            Instance of the edge class.
        """

        # initialize output layer
        kwargs.update({"n_in": self.graph[source]["n_out"], "n_out": self.graph[target]["n_in"],
                       "weights": weights, "dtype": dtype})
        trainable = True
        if train is None:
            trainable = False
            edge = Linear(**kwargs, detach=True)
        elif train == "gd":
            edge = Linear(**kwargs, detach=False)
        elif train == "rls":
            edge = RLS(**kwargs)
        else:
            raise ValueError("Invalid option for keyword argument `train`. Please see the docstring of "
                             "`Network.add_output_layer` for valid options.")

        # add node to graph
        self.graph.add_edge(source, target, edge=edge, trainable=trainable, feedback=feedback)
        return edge

    def remove_node(self, node: Union[str, int]) -> Union[Function, NN]:
        """Removes (and returns) a node from the network.
        """
        node = self.graph[node]
        self.graph.remove_node(node)
        return node["node"]

    def remove_edge(self, source: Union[str, int], target: Union[str, int]) -> Linear:
        """Removes (and returns) an edge from the network.
        """
        edge = self.graph.remove_edge(source, target)
        return edge["edge"]

    def compile(self):

        # sort edges into feedback and feedforward edges
        fb_edges = []
        ffwd_edges = []
        for edge in self.graph.edges:
            fb = self.graph[edge[0]][edge[1]].get("feedback")
            if fb:
                fb_edges.append(edge)
            else:
                ffwd_edges.append(edge)

        # create subgraph views that contains only feedback vs. feedforward edges
        self._graph_fb = self.graph.edge_subgraph(fb_edges)
        self._graph_ffwd = self.graph.edge_subgraph(ffwd_edges)

        # make sure that only a single input node exists
        in_nodes = [n for n in self._graph_ffwd.nodes if self._graph_ffwd.in_degree(n) == 0]
        if len(in_nodes) != 1:
            raise ValueError(f"Unable to identify the input node of the Network. "
                             f"Nodes that have no input edges: {in_nodes}."
                             f"Make sure that exactly one such node without input edges exists in the network.")
        self._in_node = in_nodes.pop()

        # make sure that only a single output node exists
        out_nodes = [n for n in self._graph_ffwd.nodes if self._graph_ffwd.out_degree(n) == 0]
        if len(out_nodes) != 1:
            raise ValueError(f"Unable to identify the output node of the Network. "
                             f"Nodes that have no outgoing edges: {out_nodes}."
                             f"Make sure that exactly one such node without outgoing edges exists in the network.")
        self._out_node = out_nodes.pop()

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
        # TODO: create network workflow based on node and edge forward methods and input/output fields of nodes
        n = self._graph_ffwd[self._in_node]
        while self._graph_ffwd.out_degree(n) > 0:
            x = n["node"].forward(x)

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
        for node in self.graph:
            for p in node["node"].parameters(recurse):
                yield p
        for s, t in self.graph.edges:
            for p in self.graph[s][t]["edge"].parameters():
                yield p

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

        # initialize observer
        obs = Observer(dt=self.rnn_layer.dt, record_loss=kwargs.pop("record_loss", False), **kwargs)
        rec_vars = [self._relabel_var(v) for v in obs.recorded_rnn_variables]

        # forward input through static network
        with torch.no_grad():
            for step in range(steps):
                output = self.forward(inp_tensor[step, :])
                if step % sampling_steps == 0:
                    if verbose:
                        print(f'Progress: {step}/{steps} integration steps finished.')
                    obs.record(step, output, 0.0, [self[v] for v in rec_vars])

        return obs

    def fit(self, inputs: np.ndarray, targets: np.ndarray, method: str = "gradient_descent",
            sampling_steps: int = 100, verbose: bool = True, **kwargs) -> Observer:
        """High-level method for model parameter optimization. Allows to choose between the specific
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
        # TODO: Minimize amount of code repetition across `fit` methods.
        # TODO: Rework all training/fitting methods

        super().train()
        if method == "gradient_descent":
            return self.fit_gd(inputs, targets, sampling_steps=sampling_steps, verbose=verbose, **kwargs)
        if method == "rls":
            return self.fit_rls(inputs, targets, sampling_steps=sampling_steps, verbose=verbose, **kwargs)
        if method == "force":
            if "feedback_weights" not in kwargs:
                n_out = inputs.shape[1] if self.input_layer is None else self.input_layer.weights.shape[0]
                n_in = targets.shape[1]
                kwargs["feedback_weights"] = torch.randn(n_out, n_in, device=self.device, dtype=targets.dtype)
            return self.fit_rls(inputs, targets, sampling_steps=sampling_steps, verbose=verbose, **kwargs)
        else:
            raise ValueError("Invalid training method. Please see the docstring of `Network.train` for valid choices "
                             "of the keyword argument 'method'.")

    def fit_gd(self, inputs: np.ndarray, targets: np.ndarray, optimizer: str = 'sgd', optimizer_kwargs: dict = None,
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

        # initialize loss function
        loss = self._get_loss_function(loss, loss_kwargs=loss_kwargs)

        # initialize optimizer
        optimizer = self._get_optimizer(optimizer, lr, self.parameters(), optimizer_kwargs=optimizer_kwargs)

        # retrieve keyword arguments for optimization
        step_kwargs = retrieve_from_dict(['closure'], kwargs)
        error_kwargs = retrieve_from_dict(['retain_graph'], kwargs)

        # initialize observer
        obs_kwargs = retrieve_from_dict(['record_output', 'record_loss', 'record_vars'], kwargs)
        obs = Observer(dt=self.dt, **obs_kwargs)
        rec_vars = [self._relabel_var(v) for v in obs.recorded_rnn_variables]

        # optimization
        ##############

        t0 = perf_counter()
        if len(inp_tensor.shape) > 2:
            self._train_gd_epochs(inp_tensor, target_tensor, loss, optimizer, obs, rec_vars, error_kwargs, step_kwargs,
                                  sampling_steps=sampling_steps, optim_steps=update_steps, verbose=verbose)
        else:
            self._train_gd(inp_tensor, target_tensor, loss, optimizer, obs, rec_vars, error_kwargs, step_kwargs,
                           sampling_steps=sampling_steps, optim_steps=update_steps, verbose=verbose)
        t1 = perf_counter()
        print(f'Finished optimization after {t1-t0} s.')
        return obs

    def fit_rls(self, inputs: np.ndarray, targets: np.ndarray, feedback_weights: np.ndarray = None,
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
            self.add_edge(n=targets.shape[1], train="rls", **kwargs)
        if self._model is None:
            self.compile()

        # initialize observer
        obs_kwargs = retrieve_from_dict(['record_output', 'record_loss', 'record_vars'], kwargs)
        obs = Observer(dt=self.dt, **obs_kwargs)
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

    def fit_rl(self, inputs: np.ndarray, targets: np.ndarray, feedback_weights: np.ndarray = None,
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
            self.add_edge(n=targets.shape[1], train="rls", **kwargs)

        # initialize observer
        obs_kwargs = retrieve_from_dict(['record_output', 'record_loss', 'record_vars'], kwargs)
        obs = Observer(dt=self.dt, **obs_kwargs)
        rec_vars = [self._relabel_var(v) for v in obs.recorded_rnn_variables]

        # optimization
        ##############

        t0 = perf_counter()
        if feedback_weights is None:
            feedback_weights = np.random.randn(self.rnn_layer.n_out, self.rnn_layer.n_out)
        W_fb = torch.tensor(feedback_weights, device=self.device)
        if hasattr(self._model[-1], "loss"):
            obs = self._train_rl_rls(inp_tensor, target_tensor, W_fb, epsilon, delta, obs, rec_vars,
                                     update_steps=update_steps, sampling_steps=sampling_steps, verbose=verbose,
                                     **kwargs)
        else:
            obs = self._train_rl(inp_tensor, target_tensor, W_fb, epsilon, delta, obs, rec_vars,
                                 update_steps=update_steps, sampling_steps=sampling_steps, verbose=verbose, **kwargs)
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

    def _add_layer(self, layer: Union[Linear, ActivationFunc, NN]) -> Union[Linear, ActivationFunc, NN]:
        if layer.n_in != self.n_out:
            raise ValueError("Input dimensionality of the new layer does not match the output dimensionality of the "
                             "current layer stack of the network.")
        self.graph.append(layer)
        return layer

    def _train_gd(self, inp: torch.Tensor, target: torch.Tensor, loss: Callable, optimizer: torch.optim.Optimizer,
                  obs: Observer, rec_vars: list, error_kwargs: dict, step_kwargs: dict, sampling_steps: int = 100,
                  optim_steps: int = 1, verbose: bool = False) -> Observer:

        steps = inp.shape[0]
        for step in range(steps):

            # forward pass
            prediction = self.forward(inp[step, :])

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

    def _train_gd_epochs(self, inp: torch.Tensor, target: torch.Tensor, loss: Callable,
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
                prediction = self.forward(inp[epoch, step, :])

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

    def _train_rl_rls(self, inp: torch.Tensor, targets: torch.Tensor, W_fb: torch.Tensor, epsilon: float, delta: float,
                      obs: Observer, rec_vars: list, update_steps: int = 1, sampling_steps: int = 100,
                      fb_update_steps: int = 100, verbose: bool = False, tol: float = 1e-3, loss_beta: float = 0.9,
                      noise: float = 1e-2) -> Observer:

        out_layer = self._model.pop(-1)
        steps = inp.shape[0]

        # extract properties of feedback weights
        sr_0 = torch.max(torch.abs(torch.linalg.eigvals(W_fb)))
        mask = 1.0 * (torch.abs(W_fb) > 0)
        fb_size = W_fb.shape

        # get initial step done prior to the loop
        x = self.forward(inp[0, :])
        y_hat = out_layer.forward(x)
        obs.record(0, y_hat, out_layer.loss, [self[v] for v in rec_vars])
        loss = 1.0
        loss_rl = 1.0
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
                loss_rl = 1.0/(1.0 + torch.exp(- epsilon*loss_rl - (1-epsilon)*loss_tmp))
                if step+1 % fb_update_steps == 0:
                    W_new = (loss_rl*torch.randn(fb_size, device=self.device)*noise + (1-loss_rl)*out_layer.P) * mask
                    W_fb = delta*W_fb + (1-delta)*W_new
                    sr_1 = torch.max(torch.abs(torch.linalg.eigvals(W_fb)))
                    W_fb *= sr_0/sr_1

                # results storage
                if step % sampling_steps == 0:
                    if torch.isnan(loss_tmp):
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


class FeedbackNetwork(Network):

    def __init__(self, device: str = "cpu"):

        super().__init__(device)

        # add feedback-related class attributes
        self._outputs = {}
        self.feedback = {}

    def forward(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        for i, layer in enumerate(self.graph):
            if i in self.feedback:
                feedback_layer, out_idx = self.feedback[i]
                x += feedback_layer(self._outputs[out_idx])
            x = layer(x)
            if i in self._outputs:
                self._outputs[i] = x
        return x
