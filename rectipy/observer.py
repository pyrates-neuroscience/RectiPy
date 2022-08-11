import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Iterable
from .utility import retrieve_from_dict


class Observer:

    def __init__(self, dt: float, record_output: bool = True, record_loss: bool = True, record_vars: list = None):

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

    @property
    def recorded_state_variables(self):
        return self._state_vars

    @property
    def record_loss(self):
        return self._record_loss

    @property
    def record_output(self):
        return self._record_out

    def record(self, output: torch.Tensor, loss: float, record_vars: Iterable[torch.Tensor]):
        recs = self._recordings
        for key, val, reduce in zip(self._state_vars, record_vars, self._reduce_vars):
            v = val.detach().numpy()
            recs[key].append(np.mean(v) if reduce else v)
        if self._record_out:
            recs['out'].append(output.detach().numpy())
        if self._record_loss:
            recs['loss'].append(loss)

    def plot(self, y: str, x: str = None, ax: plt.Axes = None, **kwargs):

        if ax is None:
            subplot_kwargs = retrieve_from_dict(['figsize'], kwargs)
            _, ax = plt.subplots(**subplot_kwargs)

        y_sig = np.asarray(self._recordings[y])
        x_sig = np.arange(0, y_sig.shape[0])*self._dt if x is None else self._recordings[x]

        ax.plot(x_sig, y_sig, **kwargs)
        ax.set_xlabel('time' if x is None else x)
        ax.set_ylabel(y)

        return ax
