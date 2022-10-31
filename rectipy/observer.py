import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Iterable
from pandas import DataFrame
from .utility import retrieve_from_dict


class Observer:
    """Class that is used to record state variables, outputs, and losses during calls of `Network.train`,
    `Network.test`, or `Network.run`.
    """

    def __init__(self, dt: float, record_output: bool = True, record_loss: bool = True, record_vars: list = None):
        """Instantiates observer.

        Parameters
        ----------
        dt
            Step-size of training/testing/integration steps.
        record_output
            If true, the output of the `Network` instance is recorded.
        record_loss
            If true, the loss calculated during training/testing by the `Network` is recorded.
        record_vars
            Additional variables of the RNN layer that should be recorded.
        """

        if not record_vars:
            record_vars = []
        self._dt = dt
        self._state_vars = [v[0] for v in record_vars]
        self._reduce_vars = [v[1] for v in record_vars]
        self._recordings = {v: [] for v in self._state_vars}
        self._record_loss = record_loss
        self._record_out = record_output
        if record_loss:
            self._recordings['loss'] = []
        if record_output:
            self._recordings['out'] = []
        self._recordings["steps"] = []

    def __getitem__(self, item: str):
        return DataFrame(index=self._recordings["steps"], data=self._recordings[item])

    @property
    def recorded_rnn_variables(self) -> list:
        """RNN state variables that are recorded by this `Observer` instance.
        """
        return self._state_vars

    @property
    def recordings(self):
        columns = self._state_vars
        if self._record_out:
            columns.append("out")
        if self._record_loss:
            columns.append("loss")
        data = np.asarray([self[v] for v in columns]).T
        return DataFrame(index=self._recordings["step"], data=data, columns=columns)

    def record(self, step: int, output: torch.Tensor, loss: float, record_vars: Iterable[torch.Tensor]) -> None:
        """Performs a single recording steps.

        Parameters
        ----------
        step
            Integration step.
        output
            Output of the `Network` model.
        loss
            Current loss of the `Network` model.
        record_vars
            Additional variables of the RNN layer that should be recorded.

        Returns
        -------
        None
        """
        recs = self._recordings
        recs["steps"].append(step)
        for key, val, reduce in zip(self._state_vars, record_vars, self._reduce_vars):
            v = val.detach().numpy()
            recs[key].append(np.mean(v) if reduce else v)
        if self._record_out:
            recs['out'].append(output.detach().numpy())
        if self._record_loss:
            recs['loss'].append(loss)

    def plot(self, y: str, x: str = None, ax: plt.Axes = None, **kwargs) -> plt.Axes:
        """Create a line plot with variable `y` on the y-axis and `x` on the x-axis.

        Parameters
        ----------
        y
            Name of the variable to be plotted on the y-axis.
        x
            Name of the variable to be plotted on the x-axis. If not provided, `y` will be plotted against time steps.
        ax
            `matplotlib.pyplot.Axes` instance in which to plot.
        kwargs
            Additional keyword arguments for the `matplotlib.pyplot.plot` call.

        Returns
        -------
        plt.Axes
            Instance of `matplotlib.pyplot.Axes` that contains the line plot.
        """

        if ax is None:
            subplot_kwargs = retrieve_from_dict(['figsize'], kwargs)
            _, ax = plt.subplots(**subplot_kwargs)

        y_sig = np.asarray(self._recordings[y])
        x_sig = np.asarray(self._recordings["steps"])*self._dt if x is None else self._recordings[x]

        ax.plot(x_sig, y_sig, **kwargs)
        ax.set_xlabel('time' if x is None else x)
        ax.set_ylabel(y)

        return ax

    def matshow(self, v: str, ax: plt.Axes = None, **kwargs) -> plt.Axes:
        """Create a 2D color plot of variable `v`.

        Parameters
        ----------
        v
            Name of the variable to be plotted.
        ax
            `matplotlib.pyplot.Axes` instance in which to plot.
        kwargs
            Additional keyword arguments for the `matplotlib.pyplot.imshow` call.

        Returns
        -------
        plt.Axes
            Instance of `matplotlib.pyplot.Axes` that contains the line plot.
        """

        if ax is None:
            subplot_kwargs = retrieve_from_dict(['figsize'], kwargs)
            _, ax = plt.subplots(**subplot_kwargs)

        sig = np.asarray(self._recordings[v])

        shrink = kwargs.pop("shrink", 0.6)
        im = ax.imshow(sig.T, **kwargs)
        plt.colorbar(im, ax=ax, shrink=shrink)
        ax.set_xlabel('time')
        ax.set_ylabel(v)

        return ax
