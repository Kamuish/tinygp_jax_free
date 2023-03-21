# -*- coding: utf-8 -*-
"""
This submodule defines a set of distance metrics that can be used when working
with multivariate data. By default, all
:class:`tinygp.kernels.stationary.Stationary` kernels will use either an
:class:`L1Distance` or :class:`L2Distance`, when applied in multiple dimensions,
but it is possible to define custom metrics, as discussed in the :ref:`geometry`
tutorial.
"""

from __future__ import annotations

__all__ = ["Distance", "L1Distance", "L2Distance"]

from abc import ABCMeta, abstractmethod

import numpy as np

from src.tinygp.helpers import dataclass


class Distance(metaclass=ABCMeta):
    """An abstract base class defining a distance metric interface"""

    @abstractmethod
    def distance(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute the distance between two coordinates under this metric"""
        raise NotImplementedError()

    def squared_distance(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute the squared distance between two coordinates

        By default this returns the squared result of
        :func:`tinygp.kernels.stationary.Distance.distance`, but some metrics
        can take advantage of these separate implementations to avoid
        unnecessary square roots.
        """
        return np.square(self.distance(X1, X2))


@dataclass
class L1Distance(Distance):
    """The L1 or Manhattan distance between two coordinates"""

    def distance(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return np.sum(np.abs(X1 - X2))


@dataclass
class L2Distance(Distance):
    """The L2 or Euclidean distance between two coordinates"""

    def distance(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        r1 = L1Distance().distance(X1, X2)
        r2 = self.squared_distance(X1, X2)
        zeros = np.equal(r2, 0)
        r2 = np.where(zeros, np.ones_like(r2), r2)
        return np.where(zeros, r1, np.sqrt(r2))

    def squared_distance(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return np.sum(np.square(X1 - X2))
