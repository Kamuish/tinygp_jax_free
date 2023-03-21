# -*- coding: utf-8 -*-
"""
In ``tinygp``, "transforms" are a powerful and relatively safe way to build
extremely expressive kernels without resorting to writing a fully fledged custom
kernel. More details can be found in the :ref:`transforms` tutorial.
"""

from __future__ import annotations

__all__ = ["Transform", "Linear", "Cholesky", "Subspace"]

from functools import partial
from typing import Any, Callable, Sequence, Union

from scipy import linalg

from src.tinygp.helpers import dataclass
from src.tinygp.kernels.base import Kernel


@dataclass
class Transform(Kernel):
    """Apply a transformation to the input coordinates of the kernel

    Args:
        transform: (Callable): A callable object that accepts coordinates as
            inputs and returns transformed coordinates.
        kernel (Kernel): The kernel to use in the transformed space.
    """

    transform: Callable[[Any], Any]
    kernel: Kernel

    def evaluate(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return self.kernel.evaluate(self.transform(X1), self.transform(X2))


@dataclass
class Linear(Kernel):
    """Apply a linear transformation to the input coordinates of the kernel

    For example, the following transformed kernels are all equivalent, but the
    second supports more flexible transformations:

    .. code-block:: python

        >>> import numpy as np
        >>> from tinygp import kernels, transforms
        >>> kernel0 = kernels.Matern32(4.5)
        >>> kernel1 = transforms.Linear(1.0 / 4.5, kernels.Matern32())
        >>> np.testing.assert_allclose(
        ...     kernel0.evaluate(0.5, 0.1), kernel1.evaluate(0.5, 0.1)
        ... )

    Args:
        scale (np.ndarray): A 0-, 1-, or 2-dimensional array specifying the
            scale of this transform.
        kernel (Kernel): The kernel to use in the transformed space.
    """

    scale: np.ndarray
    kernel: Kernel

    def evaluate(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        if np.ndim(self.scale) < 2:
            transform = partial(np.multiply, self.scale)
        elif np.ndim(self.scale) == 2:
            transform = partial(np.dot, self.scale)
        else:
            raise ValueError("'scale' must be 0-, 1-, or 2-dimensional")
        return self.kernel.evaluate(transform(X1), transform(X2))


@dataclass
class Cholesky(Kernel):
    """Apply a Cholesky transformation to the input coordinates of the kernel

    For example, the following transformed kernels are all equivalent, but the
    second supports more flexible transformations:

    .. code-block:: python

        >>> import numpy as np
        >>> from tinygp import kernels, transforms
        >>> kernel0 = kernels.Matern32(4.5)
        >>> kernel1 = transforms.Cholesky(4.5, kernels.Matern32())
        >>> np.testing.assert_allclose(
        ...     kernel0.evaluate(0.5, 0.1), kernel1.evaluate(0.5, 0.1)
        ... )

    Args:
        factor (np.ndarray): A 0-, 1-, or 2-dimensional array specifying the
            Cholesky factor. If 2-dimensional, this must be a lower
            triangular matrix, but this is not checked.
        kernel (Kernel): The kernel to use in the transformed space.
    """

    factor: np.ndarray
    kernel: Kernel

    def evaluate(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        if np.ndim(self.factor) < 2:
            transform = partial(np.multiply, 1.0 / self.factor)
        elif np.ndim(self.factor) == 2:
            transform = partial(
                linalg.solve_triangular, self.factor, lower=True
            )
        else:
            raise ValueError("'scale' must be 0-, 1-, or 2-dimensional")
        return self.kernel.evaluate(transform(X1), transform(X2))

    @classmethod
    def from_parameters(
        cls, diagonal: np.ndarray, off_diagonal: np.ndarray, kernel: Kernel
    ) -> "Cholesky":
        """Build a Cholesky transform with a sensible parameterization

        Args:
            diagonal (np.ndarray): An ``(ndim,)`` array with the diagonal
                elements of ``factor``. These must be positive, but this
                is not checked.
            off_diagonal (np.ndarray): An ``((ndim - 1) * ndim,)`` array
                with the off-diagonal elements of ``factor``.
            kernel (Kernel): The kernel to use in the transformed space.
        """
        ndim = diagonal.size
        if off_diagonal.size != ((ndim - 1) * ndim) // 2:
            raise ValueError(
                "Dimension mismatch: expected "
                f"(ndim-1)*ndim/2 = {((ndim - 1) * ndim) // 2} elements in "
                f"'off_diagonal'; got {off_diagonal.size}"
            )
        factor = np.zeros((ndim, ndim))
        factor = factor.at[np.diag_indices(ndim)].add(diagonal)
        factor = factor.at[np.tril_indices(ndim, -1)].add(off_diagonal)
        return cls(factor, kernel)


@dataclass
class Subspace(Kernel):
    """A kernel transform that selects a subset of the input dimensions

    For example, the following kernel only depends on the coordinates in the
    second (`1`-th) dimension:

    .. code-block:: python

        >>> import numpy as np
        >>> from tinygp import kernels, transforms
        >>> kernel = transforms.Subspace(1, kernels.Matern32())
        >>> np.testing.assert_allclose(
        ...     kernel.evaluate(np.array([0.5, 0.1]), np.array([-0.4, 0.7])),
        ...     kernel.evaluate(np.array([100.5, 0.1]), np.array([-70.4, 0.7])),
        ... )

    Args:
        axis: (Axis, optional): An integer or tuple of integers specifying the
            axes to select.
        kernel (Kernel): The kernel to use in the transformed space.
    """

    axis: Union[Sequence[int], int]
    kernel: Kernel

    def evaluate(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return self.kernel.evaluate(X1[self.axis], X2[self.axis])
