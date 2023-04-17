import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Iterable, Union, Any, Tuple
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
        self._state_vars = [v[:2] for v in record_vars]
        self._reduce_vars = [v[2] for v in record_vars]
        self._recordings = {v: [] for v in self._state_vars}
        self._record_loss = record_loss
        self._record_out = record_output
        if record_loss:
            self._recordings['loss'] = []
        if record_output:
            self._recordings['out'] = []
        self._recordings["steps"] = []
        self._additional_storage = {}

    def __getitem__(self, item: Union[str, Tuple[str, str]]):
        try:
            return self._recordings[item]
        except KeyError:
            return self._additional_storage[item]

    @property
    def recorded_state_variables(self) -> list:
        """RNN state variables that are recorded by this `Observer` instance.
        """
        return self._state_vars

    @property
    def recorded_variables(self) -> list:
        """RNN state variables that are recorded by this `Observer` instance.
        """
        return list(self._recordings.keys())

    @property
    def recordings(self):
        columns = self._state_vars
        if self._record_out:
            columns.append("out")
        if self._record_loss:
            columns.append("loss")
        data = np.asarray([self[v] for v in columns]).T
        return DataFrame(index=np.asarray(self._recordings["steps"])*self._dt, data=data, columns=columns)

    def to_dataframe(self, item: Union[str, Tuple[str, str]]):
        try:
            data = self.to_numpy(item)
            return DataFrame(index=np.asarray(self._recordings["steps"])*self._dt, data=data)
        except KeyError:
            return self[item]

    def record(self, step: int, output: torch.Tensor, loss: Union[float, torch.Tensor],
               record_vars: Iterable[torch.Tensor]) -> None:
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
            recs[key].append(torch.mean(val) if reduce else val)
        if self._record_out:
            recs['out'].append(output)
        if self._record_loss:
            recs['loss'].append(loss)

    def save(self, key: str, val: Any):
        """Saves object on observer. Can be retrieved via `key`.

        Parameters
        ----------
        key
            Used for storage/retrieval.
        val
            Object to be stored.
        """
        self._additional_storage[key] = val

    def to_numpy(self, item: Union[str, Tuple[str, str]]) -> np.ndarray:
        try:
            val = self._recordings[item]
        except KeyError:
            val = self._additional_storage[item]
        try:
            val_numpy = np.asarray([v.detach().cpu().numpy() for v in val])
        except AttributeError as e:
            raise e
        return val_numpy

    def plot(self, y: Union[str, Tuple[str, str]], x: Union[str, Tuple[str, str]] = None, ax: plt.Axes = None,
             **kwargs) -> plt.Axes:
        """Create a line plot with variable `y` on the y-axis and `x` on the x-axis.

        Parameters
        ----------
        y
            Tuple that contains the names of the node and the node variable to be plotted on the y-axis.
        x
            Tuple that contains the names of the node and the node variable to be plotted on the x-axis.
            If not provided, `y` will be plotted against time steps.
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

        if x is None:
            ax.plot(self.to_dataframe(y), **kwargs)
        else:
            ax.plot(self.to_numpy(x), self.to_numpy(y), **kwargs)

        ax.set_xlabel('time' if x is None else f"Node: {x[0]}, variable: {x[-1]}" if type(x) is tuple else x)
        ax.set_ylabel(f"Node: {y[0]}, variable: {y[-1]}" if type(y) is tuple else y)

        return ax

    def matshow(self, v: Union[str, Tuple[str, str]], ax: plt.Axes = None, **kwargs) -> plt.Axes:
        """Create a 2D color plot of variable `v`.

        Parameters
        ----------
        v
            Tuple that contains the names of the node and the node variable to be plotted.
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

        sig = self.to_dataframe(v)
        if type(sig) is not np.ndarray:
            sig = np.asarray(sig)

        shrink = kwargs.pop("shrink", 0.6)
        im = ax.imshow(sig.T, **kwargs)
        plt.colorbar(im, ax=ax, shrink=shrink)
        ax.set_xlabel('time')
        ax.set_ylabel(f"Node: {v[0]}, variable: {v[1]}" if type(v) is tuple else v)

        return ax
