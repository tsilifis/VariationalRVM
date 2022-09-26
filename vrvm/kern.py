import numpy as np


__all__ = ["Kernel", "RBF"]


class Kernel(object):

    _input_dim = None

    _name = None

    @property
    def input_dim(self):
        return self._input_dim

    @input_dim.setter
    def input_dim(self, value):
        assert isinstance(value, int)
        assert value > 0
        self._input_dim = value

    def __init__(self, input_dim, name="Kernel"):
        """
        Initialize the object
        """
        assert isinstance(input_dim, int)
        assert input_dim > 0
        self._input_dim = input_dim
        self._name = name

    def eval(self, X, X2=None):
        raise NotImplementedError


class RBF(Kernel):
    """
    A class of Kernel type representing the squared exponential covariance kernel.
    """

    _var = None

    _lengthscale = None

    _iso = None

    _n_params = None

    @property
    def var(self):
        return self._var

    @var.setter
    def var(self, value):
        assert value > 0.0
        self._var = value

    @property
    def lengthscale(self):
        return self._lengthscale

    @lengthscale.setter
    def lengthscale(self, value):
        assert value > 0.0
        self._lengthscale = value

    def __init__(self, input_dim, variance=1.0, corr_length=1.0, name="rbf", iso=True):
        """
        Initializing the object.
        """
        super(RBF, self).__init__(input_dim, name)
        assert variance > 0
        if isinstance(corr_length, list):
            assert len(corr_length) == input_dim
            self._iso = False
            for i in range(input_dim):
                assert corre_length[i] > 0
            self._lengthscale = corr_length
        else:
            assert corr_length > 0
            self._lengthscale = [corr_length] * input_dim
            self._iso = True
        self._var = variance

        if self._iso:
            self._n_params = 2
        else:
            self._n_params = self._input_dim + 1

    def eval(self, X, Y=None):
        assert X.shape[1] == self._input_dim
        if Y is None:
            diff = np.vstack(
                [
                    (X[:, i][:, None] - X[:, i][None, :]).reshape(
                        1, X.shape[0], X.shape[0]
                    )
                    / self._lengthscale[i]
                    for i in range(X.shape[1])
                ]
            )
            diff_sq = np.sum(np.square(diff), 0)
            return self._var * np.exp(-diff_sq / 2.0)
        else:
            assert Y.shape[1] == self._input_dim
            diff = np.vstack(
                [
                    (X[:, i][:, None] - Y[:, i][None, :]).reshape(
                        1, X.shape[0], Y.shape[0]
                    )
                    / self._lengthscale[i]
                    for i in range(self._input_dim)
                ]
            )
            diff_sq = np.sum(np.square(diff), 0)
            return self._var * np.exp(-diff_sq / 2.0)
