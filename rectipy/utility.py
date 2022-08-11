import numpy as np
from scipy.stats import rv_discrete, bernoulli


# helper functions
##################


def retrieve_from_dict(keys: list, data: dict) -> dict:
    new_data = {}
    for key in keys:
        if key in data:
            new_data[key] = data.pop(key)
    return new_data


def add_op_name(op: str, var: str, new_var_names: dict) -> str:
    if '/' in var:
        return var
    new_var_names[var] = f"{op}/{var}"
    return new_var_names[var]


def _wrap(idxs: np.ndarray, N: int) -> np.ndarray:
    idxs[idxs < 0] = N+idxs[idxs < 0]
    idxs[idxs >= N] = idxs[idxs >= N] - N
    return idxs


# connectivity generation functions
###################################


def circular_connectivity(N: int, p: float, spatial_distribution: rv_discrete) -> np.ndarray:
    C = np.zeros((N, N))
    n_conns = int(N*p)
    for n in range(N):
        idxs = spatial_distribution.rvs(size=n_conns)
        signs = 1 * (bernoulli.rvs(p=0.5, loc=0, size=n_conns) > 0)
        signs[signs == 0] = -1
        conns = _wrap(n + idxs*signs, N)
        C[n, conns] = 1.0/n_conns
    return C


def random_connectivity(N: int, p: float, normalize: bool = True) -> np.ndarray:
    C = np.zeros((N, N))
    n_conns = int(N * p)
    positions = np.arange(start=0, stop=N)
    for n in range(N):
        idxs = np.random.permutation(positions)[:n_conns]
        C[n, idxs] = 1.0/n_conns if normalize else 1.0
    return C
