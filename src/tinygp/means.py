# -*- coding: utf-8 -*-
"""
In ``tinygp``, the Gaussian process mean function can be defined using any
callable object, but this submodule includes two helper classes for defining
means. When defining your own mean function, it's important to remember that
your callable should accept as input a single input coordinate (i.e. not a
*vector* of coordinates), and return the scalar value of the mean at that
coordinate. ``tinygp`` will handle all the relevant ``vmap``-ing and
broadcasting.
"""

from __future__ import annotations

__all__ = ["Mean", "Conditioned"]

from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from src.tinygp.helpers import JAXArray, dataclass



@dataclass
class Mean:
    """A wrapper for the GP mean which supports a constant value or a callable

    In ``tinygp``, a mean function can be any callable which takes as input a
    single coordinate and returns the scalar mean at that location.

    Args:
        value: Either a *scalar* constant, or a callable with the correct
            signature.
    """

    value: Union[JAXArray, Callable[[JAXArray], JAXArray]]

    if TYPE_CHECKING:

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    def __call__(self, X: JAXArray) -> JAXArray:
        if callable(self.value):
            return self.value(X)
        return self.value

