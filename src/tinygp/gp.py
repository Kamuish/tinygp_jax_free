# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = ["GaussianProcess"]

from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from src.tinygp import kernels, means
from src.tinygp.helpers import JAXArray
from src.tinygp.kernels.quasisep import Quasisep
from src.tinygp.noise import Diagonal, Noise
from src.tinygp.solvers import QuasisepSolver
from src.tinygp.solvers.quasisep.core import SymmQSM

class GaussianProcess:
    """An interface for designing a Gaussian Process regression model

    Args:
        kernel (Kernel): The kernel function
        X (JAXArray): The input coordinates. This can be any PyTree that is
            compatible with ``kernel`` where the zeroth dimension is ``N_data``,
            the size of the data set.
        diag (JAXArray, optional): The value to add to the diagonal of the
            covariance matrix, often used to capture measurement uncertainty.
            This should be a scalar or have the shape ``(N_data,)``. If not
            provided, this will default to the square root of machine epsilon
            for the data type being used. This can sometimes be sufficient to
            avoid numerical issues, but if you're getting NaNs, try increasing
            this value.
        noise (Noise, optional): Used to implement more expressive observation
            noise models than those supported by just ``diag``. This can be any
            object that implements the :class:`tinygp.noise.Noise` protocol. If
            this is provided, the ``diag`` parameter will be ignored.
        mean (Callable, optional): A callable or constant mean function that
            will be evaluated with the ``X`` as input: ``mean(X)``
        solver: The solver type to be used to execute the required linear
            algebra.
    """

    def __init__(
        self,
        kernel: kernels.Kernel,
        X: np.ndarray,
        mean,
        diag: Optional[np.ndarray] = None,
        noise: Optional[Noise] = None,
        solver: Optional[Any] = None,
        mean_value: Optional[np.ndarray] = None,
        covariance_value: Optional[Any] = None,
        **solver_kwargs: Any,
    ):
        self.kernel = kernel
        self.X = X

        if callable(mean):
            self.mean_function = mean
        else:
            if mean is None:
                self.mean_function = means.Mean(jnp.zeros(()))
            else:
                self.mean_function = means.Mean(mean)
        if mean_value is None:
            mean_value = np.asarray(list(map(self.mean_function, self.X)))

        self.num_data = mean_value.shape[0]
        self.dtype = mean_value.dtype
        self.loc = self.mean = mean_value
        if self.mean.ndim != 1:
            raise ValueError(
                "Invalid mean shape: "
                f"expected ndim = 1, got ndim={self.mean.ndim}"
            )

        if noise is None:
            diag = _default_diag(self.mean) if diag is None else diag
            noise = Diagonal(diag=np.broadcast_to(diag, self.mean.shape))
        self.noise = noise

        if solver is None:
            if isinstance(covariance_value, SymmQSM) or isinstance(
                kernel, Quasisep
            ):
                solver = QuasisepSolver
            else:
                solver = DirectSolver
        self.solver = solver.init(
            kernel,
            self.X,
            self.noise,
            covariance=covariance_value,
            **solver_kwargs,
        )

    @property
    def variance(self) -> JAXArray:
        return self.solver.variance()

    @property
    def covariance(self) -> JAXArray:
        return self.solver.covariance()

    def log_probability(self, y: JAXArray) -> JAXArray:
        """Compute the log probability of this multivariate normal

        Args:
            y (JAXArray): The observed data. This should have the shape
                ``(N_data,)``, where ``N_data`` was the zeroth axis of the ``X``
                data provided when instantiating this object.

        Returns:
            The marginal log probability of this multivariate normal model,
            evaluated at ``y``.
        """
        return self._compute_log_prob(self._get_alpha(y))

    def _compute_log_prob(self, alpha: JAXArray) -> JAXArray:
        loglike = (
            -0.5 * np.sum(np.square(alpha)) - self.solver.normalization()
        )
        return np.where(np.isfinite(loglike), loglike, -np.inf)

    def _get_alpha(self, y: JAXArray) -> JAXArray:
        return self.solver.solve_triangular(y - self.loc)


def _default_diag(reference: JAXArray) -> JAXArray:
    """Default to adding some amount of jitter to the diagonal, just in case,
    we use sqrt(eps) for the dtype of the mean function because that seems to
    give sensible results in general.
    """
    return np.sqrt(np.finfo(reference).eps)
