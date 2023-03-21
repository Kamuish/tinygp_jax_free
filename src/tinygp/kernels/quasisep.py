# -*- coding: utf-8 -*-
"""
The kernels implemented in this subpackage are used with the
:class:`tinygp.solvers.QuasisepSolver` to allow scalable GP computations by
exploting quasiseparable structure in the relevant matrices (see
:ref:`api-solvers-quasisep` for more technical details). For now, these methods
are experimental, so you may find the documentation patchy in places. You are
encouraged to `open issues or pull requests
<https://github.com/dfm/tinygp/issues>`_ as you find gaps.
"""

from __future__ import annotations

__all__ = [
    "Quasisep",
    "Wrapper",
    "Sum",
    "Product",
    "Scale",
    "SHO",
    "Exp",
    "Matern32",
    "Matern52",
    "Cosine",
    "CARMA",
]

from abc import ABCMeta, abstractmethod
from typing import Any, Optional, Union

import numpy as np
import scipy as sp
import numpy as np

from src.tinygp.helpers import JAXArray, dataclass, field
from src.tinygp.kernels.base import Kernel
from src.tinygp.solvers.quasisep.core import DiagQSM, StrictLowerTriQSM, SymmQSM
from src.tinygp.solvers.quasisep.general import GeneralQSM


class Quasisep(Kernel, metaclass=ABCMeta):
    """The base class for all quasiseparable kernels

    Instead of directly implementing the ``p``, ``q``, and ``a`` elements of the
    :class:`tinygp.solvers.quasisep.core.StrictLowerQSM`, this class implements
    ``h``, ``Pinf``, and ``A``, where:

    - ``q = h``,
    - ``p = h.T @ Pinf @ A``, and
    - ``a = A``.

    This notation follows the notation from state space models for stochastic
    differential equations, and so far it seems like a good way to specify these
    models, but these details are subject to change in future versions of
    ``tinygp``.
    """

    @abstractmethod
    def design_matrix(self) :
        """The design matrix for the process"""
        raise NotImplementedError

    @abstractmethod
    def stationary_covariance(self) :
        """The stationary covariance of the process"""
        raise NotImplementedError

    @abstractmethod
    def observation_model(self, X) :
        """The observation model for the process"""
        raise NotImplementedError

    @abstractmethod
    def transition_matrix(self, X1, X2) :
        """The transition matrix between two coordinates"""
        raise NotImplementedError

    def coord_to_sortable(self, X) :
        """A helper function used to convert coordinates to sortable 1-D values

        By default, this is the identity, but in cases where ``X`` is structured
        (e.g. multivariate inputs), this can be used to appropriately unwrap
        that structure.
        """
        return X

    def to_symm_qsm(self, X) -> SymmQSM:
        """The symmetric quasiseparable representation of this kernel"""
        Pinf = self.stationary_covariance()
        xi = np.append([X[:1]], X[:-1])
        a = np.asarray(list(self.transition_matrix(i, j) for i, j in zip(xi, X)))

        h = np.asarray(list(map(self.observation_model, X)))
        q = h
        p = h @ Pinf
        d = np.sum(p * q, axis=1)
        p = np.asarray([np.dot(i, j) for i, j in zip(p, a)]) # TODO: this one

        # print("Function:: to_symm_qsm")
        # print("It is this shape: ", p.shape)
        # print("X values", X)
        # print("a values: ", a, a.shape)
        # print("----------///----")
        # print("P values: ", p)
        return SymmQSM(
            diag=DiagQSM(d=d), lower=StrictLowerTriQSM(p=p, q=q, a=a)
        )

    def to_general_qsm(self, X1, X2) -> GeneralQSM:
        """The generalized quasiseparable representation of this kernel"""

        # print("Function:: to_general_qsm")
        raise ValueError
        idx = np.searchsorted(self.coord_to_sortable(X2), self.coord_to_sortable(X1), side="right") - 1

        Xs = np.append([X2[:1]], X2[:-1])
        Pinf = self.stationary_covariance()
        a = self.transition_matrix(Xs, X2)
        h1 = np.vectorize(self.observation_model,X1)
        h2 = np.vectorize(self.observation_model, X2)

        ql = h2
        pl = h1 @ Pinf
        qu = h1
        pu = h2 @ Pinf

        i = np.clip(idx, 0, ql.shape[0] - 1)
        Xi = list(map(lambda x: np.asarray(x)[i], X2))
        pl = np.dot(pl, self.transition_matrix(Xi, X1))

        i = np.clip(idx + 1, 0, pu.shape[0] - 1)
        Xi = list(map(lambda x: np.asarray(x)[i], X2))
        qu = np.dot(self.transition_matrix(X1, Xi), qu)

        return GeneralQSM(pl=pl, ql=ql, pu=pu, qu=qu, a=a, idx=idx)

    def matmul(
        self,
        X1,
        X2: Optional[JAXArray] = None,
        y: Optional[JAXArray] = None,
    ) :
        if y is None:
            assert X2 is not None
            y = X2
            X2 = None

        if X2 is None:
            return self.to_symm_qsm(X1) @ y

        else:
            return self.to_general_qsm(X1, X2) @ y

    def __add__(self, other: Union["Kernel", JAXArray]) -> "Kernel":
        if not isinstance(other, Quasisep):
            raise ValueError(
                "Quasisep kernels can only be added to other Quasisep kernels"
            )
        return Sum(self, other)

    def __radd__(self, other: Any) -> "Kernel":
        # We'll hit this first branch when using the `sum` function
        if other == 0:
            return self
        if not isinstance(other, Quasisep):
            raise ValueError(
                "Quasisep kernels can only be added to other Quasisep kernels"
            )
        return Sum(other, self)

    def __mul__(self, other: Union["Kernel", JAXArray]) -> "Kernel":
        if isinstance(other, Quasisep):
            return Product(self, other)
        if isinstance(other, Kernel) or np.ndim(other) != 0:
            raise ValueError(
                "Quasisep kernels can only be multiplied by scalars and other "
                "Quasisep kernels"
            )
        return Scale(kernel=self, scale=other)

    def __rmul__(self, other: Any) -> "Kernel":
        if isinstance(other, Quasisep):
            return Product(other, self)
        if isinstance(other, Kernel) or np.ndim(other) != 0:
            raise ValueError(
                "Quasisep kernels can only be multiplied by scalars and other "
                "Quasisep kernels"
            )
        return Scale(kernel=self, scale=other)

    def evaluate(self, X1, X2) :
        """The kernel evaluated via the quasiseparable representation"""
        Pinf = self.stationary_covariance()
        h1 = self.observation_model(X1)
        h2 = self.observation_model(X2)
        return np.where(
            self.coord_to_sortable(X1) < self.coord_to_sortable(X2),
            h2 @ Pinf @ self.transition_matrix(X1, X2) @ h1,
            h1 @ Pinf @ self.transition_matrix(X2, X1) @ h2,
        )

    def evaluate_diag(self, X) :
        """For quasiseparable kernels, the variance is simple to compute"""
        h = self.observation_model(X)
        return h @ self.stationary_covariance() @ h


@dataclass
class Wrapper(Quasisep, metaclass=ABCMeta):
    """A base class for wrapping kernels with some custom implementations"""

    kernel: Quasisep

    def coord_to_sortable(self, X) :
        return self.kernel.coord_to_sortable(X)

    def design_matrix(self) :
        return self.kernel.design_matrix()

    def stationary_covariance(self) :
        return self.kernel.stationary_covariance()

    def observation_model(self, X) :
        return self.kernel.observation_model(self.coord_to_sortable(X))

    def transition_matrix(self, X1, X2) :
        return self.kernel.transition_matrix(
            self.coord_to_sortable(X1), self.coord_to_sortable(X2)
        )


@dataclass
class Sum(Quasisep):
    """A helper to represent the sum of two quasiseparable kernels"""

    kernel1: Quasisep
    kernel2: Quasisep

    def coord_to_sortable(self, X) :
        """We assume that both kernels use the same coordinates"""
        return self.kernel1.coord_to_sortable(X)

    def design_matrix(self) :
        return jsp.linalg.block_diag(
            self.kernel1.design_matrix(), self.kernel2.design_matrix()
        )

    def stationary_covariance(self) :
        return jsp.linalg.block_diag(
            self.kernel1.stationary_covariance(),
            self.kernel2.stationary_covariance(),
        )

    def observation_model(self, X) :
        return np.concatenate(
            (
                self.kernel1.observation_model(X),
                self.kernel2.observation_model(X),
            )
        )

    def transition_matrix(self, X1, X2) :
        return jsp.linalg.block_diag(
            self.kernel1.transition_matrix(X1, X2),
            self.kernel2.transition_matrix(X1, X2),
        )


@dataclass
class Product(Quasisep):
    """A helper to represent the product of two quasiseparable kernels"""

    kernel1: Quasisep
    kernel2: Quasisep

    def coord_to_sortable(self, X) :
        """We assume that both kernels use the same coordinates"""
        return self.kernel1.coord_to_sortable(X)

    def design_matrix(self) :
        F1 = self.kernel1.design_matrix()
        F2 = self.kernel2.design_matrix()
        return _prod_helper(F1, np.eye(F2.shape[0])) + _prod_helper(
            np.eye(F1.shape[0]), F2
        )

    def stationary_covariance(self) :
        return _prod_helper(
            self.kernel1.stationary_covariance(),
            self.kernel2.stationary_covariance(),
        )

    def observation_model(self, X) :
        return _prod_helper(
            self.kernel1.observation_model(X),
            self.kernel2.observation_model(X),
        )

    def transition_matrix(self, X1, X2) :
        return _prod_helper(
            self.kernel1.transition_matrix(X1, X2),
            self.kernel2.transition_matrix(X1, X2),
        )


@dataclass
class Scale(Wrapper):
    """The product of a scalar and a quasiseparable kernel"""

    scale: np.ndarray

    def stationary_covariance(self) :
        return self.scale * self.kernel.stationary_covariance()


@dataclass
class Matern32(Quasisep):
    r"""A scalable implementation of :class:`tinygp.kernels.stationary.Matern32`

    This kernel takes the form:

    .. math::

        k(\tau)=\sigma^2\,\left(1+f\,\tau\right)\,\exp(-f\,\tau)

    for :math:`\tau = |x_i - x_j|` and :math:`f = \sqrt{3} / \ell`.

    Args:
        scale: The parameter :math:`\ell`.
        sigma: The parameter :math:`\sigma`.
    """
    scale: np.ndarray
    sigma: np.ndarray = field(default_factory=lambda: np.ones(()))

    def noise(self) :
        f = np.sqrt(3) / self.scale
        return 4 * f**3

    def design_matrix(self) :
        f = np.sqrt(3) / self.scale
        return np.array([[0, 1], [-np.square(f), -2 * f]])

    def stationary_covariance(self) :
        return np.diag(np.array([1, 3 / np.square(self.scale)]))

    def observation_model(self, X) :
        return np.array([self.sigma, 0])

    def transition_matrix(self, X1, X2) :
        dt = X2 - X1
        f = np.sqrt(3) / self.scale
        return np.exp(-f * dt) * np.array(
            [[1 + f * dt, -np.square(f) * dt], [dt, 1 - f * dt]]
        )


@dataclass
class Matern52(Quasisep):
    r"""A scalable implementation of :class:`tinygp.kernels.stationary.Matern52`

    This kernel takes the form:

    .. math::

        k(\tau)=\sigma^2\,\left(1+f\,\tau + \frac{f^2\,\tau^2}{3}\right)
            \,\exp(-f\,\tau)

    for :math:`\tau = |x_i - x_j|` and :math:`f = \sqrt{5} / \ell`.

    Args:
        scale: The parameter :math:`\ell`.
        sigma: The parameter :math:`\sigma`.
    """

    scale: np.ndarray
    sigma: np.ndarray = field(default_factory=lambda: np.ones(()))

    def design_matrix(self) :
        f = np.sqrt(5) / self.scale
        f2 = np.square(f)
        return np.array([[0, 1, 0], [0, 0, 1], [-f2 * f, -3 * f2, -3 * f]])

    def stationary_covariance(self) :
        f = np.sqrt(5) / self.scale
        f2 = np.square(f)
        f2o3 = f2 / 3
        return np.array(
            [[1, 0, -f2o3], [0, f2o3, 0], [-f2o3, 0, np.square(f2)]]
        )

    def observation_model(self, X) :
        return np.array([self.sigma, 0, 0])

    def transition_matrix(self, X1, X2) :
        dt = X2 - X1
        f = np.sqrt(5) / self.scale
        f2 = np.square(f)
        d2 = np.square(dt)
        return np.exp(-f * dt) * np.array(
            [
                [
                    0.5 * f2 * d2 + f * dt + 1,
                    -0.5 * f * f2 * d2,
                    0.5 * f2 * f * dt * (f * dt - 2),
                ],
                [
                    dt * (f * dt + 1),
                    -f2 * d2 + f * dt + 1,
                    f2 * dt * (f * dt - 3),
                ],
                [
                    0.5 * d2,
                    0.5 * dt * (2 - f * dt),
                    0.5 * f2 * d2 - 2 * f * dt + 1,
                ],
            ]
        )


def _prod_helper(a1, a2) :
    i, j = np.meshgrid(np.arange(a1.shape[0]), np.arange(a2.shape[0]))
    i = i.flatten()
    j = j.flatten()
    if a1.ndim == 1:
        return a1[i] * a2[j]
    elif a1.ndim == 2:
        return a1[i[:, None], i[None, :]] * a2[j[:, None], j[None, :]]
    else:
        raise NotImplementedError
