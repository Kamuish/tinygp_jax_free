# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = [
    "Kernel",
    "Conditioned",
    "Custom",
    "Sum",
    "Product",
    "Constant",
    "DotProduct",
    "Polynomial",
]

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, Union

import numpy as np

from src.tinygp.helpers import dataclass, field



Axis = Union[int, Sequence[int]]


class Kernel(metaclass=ABCMeta):
    """The base class for all kernel implementations

    This subclass provides default implementations to add and multiply kernels.
    Subclasses should accept parameters in their ``__init__`` and then override
    :func:`Kernel.evaluate` with custom behavior.
    """

    if TYPE_CHECKING:

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    @abstractmethod
    def evaluate(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Evaluate the kernel at a pair of input coordinates

        This should be overridden be subclasses to return the kernel-specific
        value. Two things to note:

        1. Users shouldn't generally call :func:`Kernel.evaluate`. Instead,
           always "call" the kernel instance directly; for example, you can
           evaluate the Matern-3/2 kernel using ``Matern32(1.5)(x1, x2)``, for
           arrays of input coordinates ``x1`` and ``x2``.
        2. When implementing a custom kernel, this method should treat ``X1``
           and ``X2`` as single datapoints. In other words, these inputs will
           typically either be scalars of have shape ``n_dim``, where ``n_dim``
           is the number of input dimensions, rather than ``n_data`` or
           ``(n_data, n_dim)``, and you should let the :class:`Kernel` ``vmap``
           magic handle all the broadcasting for you.
        """
        raise NotImplementedError

    def evaluate_diag(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the kernel on its diagonal

        The default implementation simply calls :func:`Kernel.evaluate` with
        ``X`` as both arguments, but subclasses can use this to make diagonal
        calcuations more efficient.
        """
        return self.evaluate(X, X)

    def matmul(
        self,
        X1: np.ndarray,
        X2: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if y is None:
            assert X2 is not None
            y = X2
            X2 = None

        if X2 is None:
            X2 = X1

        return np.dot(self(X1, X2), y)

    def __call__(
        self, X1: np.ndarray, X2: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if X2 is None:
            k = jax.vmap(self.evaluate_diag, in_axes=0)(X1)
            if k.ndim != 1:
                raise ValueError(
                    "Invalid kernel diagonal shape: "
                    f"expected ndim = 1, got ndim={k.ndim} "
                    "check the dimensions of parameters and custom kernels"
                )
            return k
        k = jax.vmap(
            jax.vmap(self.evaluate, in_axes=(None, 0)), in_axes=(0, None)
        )(X1, X2)
        if k.ndim != 2:
            raise ValueError(
                "Invalid kernel shape: "
                f"expected ndim = 2, got ndim={k.ndim} "
                "check the dimensions of parameters and custom kernels"
            )
        return k

    def __add__(self, other: Union["Kernel", np.ndarray]) -> "Kernel":
        if isinstance(other, Kernel):
            return Sum(self, other)
        return Sum(self, Constant(other))

    def __radd__(self, other: Any) -> "Kernel":
        # We'll hit this first branch when using the `sum` function
        if other == 0:
            return self
        if isinstance(other, Kernel):
            return Sum(other, self)
        return Sum(Constant(other), self)

    def __mul__(self, other: Union["Kernel", np.ndarray]) -> "Kernel":
        if isinstance(other, Kernel):
            return Product(self, other)
        return Product(self, Constant(other))

    def __rmul__(self, other: Any) -> "Kernel":
        if isinstance(other, Kernel):
            return Product(other, self)
        return Product(Constant(other), self)


@dataclass
class Conditioned(Kernel):
    """A kernel used when conditioning a process on data

    Args:
        X: The coordinates of the data.
        scale_tril: The lower Cholesky factor of the base process' kernel
            matrix.
        kernel: The predictive kerenl; this will generally be the kernel from
            the kernel used by the original process.
    """

    X: np.ndarray
    solver: Solver
    kernel: Kernel

    def evaluate(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        kernel_vec = jax.vmap(self.kernel.evaluate, in_axes=(0, None))
        K1 = self.solver.solve_triangular(kernel_vec(self.X, X1))
        K2 = self.solver.solve_triangular(kernel_vec(self.X, X2))
        return self.kernel.evaluate(X1, X2) - K1.transpose() @ K2

    def evaluate_diag(self, X: np.ndarray) -> np.ndarray:
        kernel_vec = jax.vmap(self.kernel.evaluate, in_axes=(0, None))
        K = self.solver.solve_triangular(kernel_vec(self.X, X))
        return self.kernel.evaluate_diag(X) - K.transpose() @ K


@dataclass
class Custom(Kernel):
    """A custom kernel class implemented as a callable

    Args:
        function: A callable with a signature and behavior that matches
            :func:`Kernel.evaluate`.
    """

    function: Callable[[Any, Any], Any]

    def evaluate(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return self.function(X1, X2)


@dataclass
class Sum(Kernel):
    """A helper to represent the sum of two kernels"""

    kernel1: Kernel
    kernel2: Kernel

    def evaluate(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return self.kernel1.evaluate(X1, X2) + self.kernel2.evaluate(X1, X2)


@dataclass
class Product(Kernel):
    """A helper to represent the product of two kernels"""

    kernel1: Kernel
    kernel2: Kernel

    def evaluate(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return self.kernel1.evaluate(X1, X2) * self.kernel2.evaluate(X1, X2)


@dataclass
class Constant(Kernel):
    r"""This kernel returns the constant

    .. math::

        k(\mathbf{x}_i,\,\mathbf{x}_j) = c

    where :math:`c` is a parameter.

    Args:
        c: The parameter :math:`c` in the above equation.
    """

    value: np.ndarray

    def evaluate(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        if np.ndim(self.value) != 0:
            raise ValueError("The value of a constant kernel must be a scalar")
        return self.value

