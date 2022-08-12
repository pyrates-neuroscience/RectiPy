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


def random_connectivity(n: int, m: int, p: float, normalize: bool = True) -> np.ndarray:
    C = np.zeros((n, m))
    n_conns = int(m*p)
    positions = np.arange(start=0, stop=m)
    for row in range(n):
        cols = np.random.permutation(positions)[:n_conns]
        C[row, cols] = 1.0/n_conns if normalize else 1.0
    return C


def input_connections(n: int, m: int, p: float, variance: float = 1.0, zero_mean: bool = True):
    C_tmp = random_connectivity(m, n, p, normalize=False).T
    C = np.zeros_like(C_tmp)
    for col in range(C_tmp.shape[1]):
        rows = np.argwhere(C_tmp[:, col] > 0).squeeze()
        C[rows, col] = np.random.randn(rows.shape[0])*variance
        if zero_mean:
            C[rows, col] -= np.sum(C[rows, col])
    return C


# score functions
#################


def wta_score(x: np.ndarray, y: np.ndarray):
    z = np.zeros((x.shape[0]))
    for idx in range(x.shape[0]):
        z[idx] = 1.0 if np.argmax(x[idx, :]) == np.argmax(y[idx, :]) else 0.0
    return np.mean(z)
