import numpy as np
from scipy.stats import rv_discrete, bernoulli
from typing import Union


# helper functions
##################


def retrieve_from_dict(keys: list, data: dict) -> dict:
    """Remove dictionary entries and collect them in a new dictionary.

    Parameters
    ----------
    keys
        Entries in `data` that are to be removed from `data` and collected in a new dict.
    data
        Original dictionary.

    Returns
    -------
    dict
        New dictionary that contains the key-value pairs that were originally stored in `data` under `keys`.
    """
    new_data = {}
    for key in keys:
        if key in data:
            new_data[key] = data.pop(key)
    return new_data


def add_op_name(op: str, var: Union[str, None], new_var_names: dict) -> Union[str, None]:
    """Adds an operator name to a variable identifier.

    Parameters
    ----------
    op
        Operator name to be added.
    var
        Current variable identifier.
    new_var_names
        Dictionary that contains the maping between old and updated varaible identifiers.

    Returns
    -------
    str
        Updated variable name.
    """
    if var is None:
        return var
    elif "/" in var:
        _, v = var.split("/")
        new_var_names[v] = var
        return new_var_names[v]
    new_var_names[var] = f"{op}/{var}"
    return new_var_names[var]


def _wrap(idxs: np.ndarray, N: int) -> np.ndarray:
    idxs[idxs < 0] = N+idxs[idxs < 0]
    idxs[idxs >= N] = idxs[idxs >= N] - N
    return idxs


# connectivity generation functions
###################################


def circular_connectivity(N: int, p: float, spatial_distribution: rv_discrete) -> np.ndarray:
    """Generate a coupling matrix between nodes aligned on a circle.

    Parameters
    ----------
    N
        Number of nodes.
    p
        Connection probability.
    spatial_distribution
        Probability distribution defined over space. Will be used to draw indices of nodes from which each node in the
        circular network receives inputs.

    Returns
    -------
    np.ndarray
        2D coupling matrix (N x N).
    """
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
    """Generate a random coupling matrix.

    Parameters
    ----------
    n
        Number of rows
    m
        Number of columns
    p
        Coupling probability.
    normalize
        If true, all rows will be normalized such that they sum up to 1.

    Returns
    -------
    np.ndarray
        2D couping matrix (n x m).
    """
    C = np.zeros((n, m))
    n_conns = int(m*p)
    positions = np.arange(start=0, stop=m)
    for row in range(n):
        cols = np.random.permutation(positions)[:n_conns]
        C[row, cols] = 1.0/n_conns if normalize else 1.0
    return C


def input_connections(n: int, m: int, p: float, variance: float = 1.0, zero_mean: bool = True):
    """Generate random input connections.

    Parameters
    ----------
    n
        Number of rows.
    m
        Number of columns.
    p
        Coupling probability.
    variance
        Variance of the randomly drawn input weights in each row.
    zero_mean
        If true, input weights in each row will be normalized such that they sum up to 0.

    Returns
    -------
    np.ndarray
        2D input weight matrix (n x m)
    """
    C_tmp = random_connectivity(m, n, p, normalize=False).T
    C = np.zeros_like(C_tmp)
    for col in range(C_tmp.shape[1]):
        rows = np.argwhere(C_tmp[:, col] > 0).squeeze()
        C[rows, col] = np.random.randn(rows.shape[0])*variance
        if zero_mean:
            C[rows, col] -= np.sum(C[rows, col])
    return C


def normalize(x: np.ndarray, mode: str = "minmax", row_wise: bool = False) -> np.ndarray:
    """Normalization function for matrices.

    Parameters
    ----------
    x
        N x m matrix.
    mode
        Normalization mode. Can be one of the following options:
        - 'minmax': Normalize such that the minimum of the data is 0 and the maximum is 1.
        - 'zscore': Normalize data such that the mean is 0 and the standard deviation is 1.
        - 'sum': Normalize such that the sum over the data equals 1.
    row_wise
        If true, normalization will be applied independently for each row of `x`.

    Returns
    -------
    np.ndarray
        N x m matrix, normalized.
    """
    if row_wise:
        for i in range(x.shape[0]):
            x[i, :] = normalize(x[i, :], mode=mode, row_wise=False)
    else:
        x_tmp = x.flatten()
        if mode == "minmax":
            x -= np.min(x_tmp)
            max_val = np.max(x_tmp)
            if max_val > 0:
                x /= max_val
        elif mode == "zscore":
            x -= np.mean(x_tmp)
            std = np.std(x_tmp)
            if std > 0:
                x /= std
        elif mode == "sum":
            x /= np.sum(x_tmp)
        else:
            raise ValueError(f"Invalid normalization mode: {mode}.")
    return x


# function for optimization
###########################


def wta_score(x: np.ndarray, y: np.ndarray) -> float:
    """Calculates the winner-takes-all score.

    Parameters
    ----------
    x
        2D array, where rows are samples and columns are features.
    y
        2D array, where rows are samples and columns are features.

    Returns
    -------
    float
        WTA score.

    """
    z = np.zeros((x.shape[0],))
    for idx in range(x.shape[0]):
        z[idx] = 1.0 if np.argmax(x[idx, :]) == np.argmax(y[idx, :]) else 0.0
    return float(np.mean(z))


def readout(X: np.ndarray, y: np.ndarray, k: int = 1, train_split: int = None, verbose: bool = True, **kwargs
            ) -> dict:
    """Uses Ridge regression to find a set of coefficients `a` that minimizes `y - aX`.

    Parameters
    ----------
    X
        2D array of data (rows = samples, columns = features).
    y
        2D array of data (rows = samples, columns = output dimensions).
    k
        If larger 1, `k` splits into training and testing data will be performed, and for each of these splits a ridge
        regression will be calculated.
    train_split
        An integer that splits `X` and `y` into a training and a test data set. Must be smaller than the size of the
        first dimension of `X` and `y`.
    verbose
        If true, updates about the regression procedure will be displayed.
    kwargs
        Additional keyword arguments passed to `sklearn.linear_model.Ridge`

    Returns
    -------
    dict
        Contains entries with the following keys:
         - 'train_score': R2 score training data
         - 'test_score': R2 score on test data
         - 'readout_weights': Regression coefficients
         - 'readout_bias': Regression bias term
         - 'prediction': Prediction of the fitted classifier for the testing data
         - 'target': Prediction target, i.e. the testing data
    """

    # imports
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import StratifiedKFold

    # split into training and test data set
    if train_split:
        X, X_t = X[:train_split], X[train_split:]
        y, y_t = y[:train_split], y[train_split:]

    # perform ridge regression
    if k > 1:
        splitter = StratifiedKFold(n_splits=k)
        scores, coefs, intercepts = [], [], []
        for i, (train_idx, test_idx) in enumerate(splitter.split(X=X, y=y)):
            classifier = Ridge(**kwargs)
            classifier.fit(X[train_idx], y[train_idx])
            scores.append(classifier.score(X=X[test_idx], y=y[test_idx]))
            coefs.append(classifier.coef_)
            intercepts.append(classifier.intercept_)
    else:
        classifier = Ridge(**kwargs)
        classifier.fit(X, y)
        scores = [classifier.score(X=X, y=y)]
        coefs = [classifier.coef_]
        intercepts = [classifier.intercept_]

    # store readout weights
    w_out = np.mean(coefs, axis=0)
    intercept = np.mean(intercepts, axis=0)
    train_score = np.mean(scores)

    if verbose:
        print(f'Finished readout training.')
        if k > 1:
            print(fr'Average, cross-validated $R^2$ score across {k} test folds: {train_score}')
        else:
            print(fr'$R^2$ score on training data: {train_score}')

    # store temporary results
    res = {"train_score": train_score, "readout_weights": w_out, "readout_bias": intercept, "test_score": None,
           "prediction": None, "target": None}

    # perform testing
    if train_split:

        classifier = Ridge(**kwargs)
        classifier.coef_ = w_out
        classifier.intercept_ = intercept

        res["test_score"] = classifier.score(X=X_t, y=y_t)
        res["prediction"] = classifier.predict(X_t)
        res["target"] = y_t

        if verbose:
            print(f'Finished readout training.')
            print(fr"$R^2$ score on testing data: {res['test_score']}")

    else:

        res["prediction"] = classifier.predict(X)
        res["target"] = y

    return res
