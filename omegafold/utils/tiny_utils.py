# -*- coding: utf-8 -*-
# =============================================================================
# Copyright 2022 HeliXon Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""
Tinygrad utilities
"""
# =============================================================================
# Imports
# =============================================================================
import numbers
import typing

import tinygrad
from tinygrad.nn import LayerNorm

# =============================================================================
# Constants
# =============================================================================

T = typing.TypeVar("T")


# =============================================================================
# Functions
# =============================================================================
def mask2bias(mask: tinygrad.Tensor, *, inf: float = 1e9) -> tinygrad.Tensor:
    """Convert mask to attention bias

    Args:
        mask: the mask to convert to bias representation
        inf: the floating point number to represent infinity

    Returns:
        bias representation for masking in attention

    """
    return mask.float().sub(1).mul(inf)


def normalize(
        inputs: tinygrad.Tensor,
        normalized_shape: typing.Optional[
            typing.Union[int, typing.List[int], typing.Tuple[int]]] = None,
        in_place: bool = False
) -> tinygrad.Tensor:
    """Layer normalization without a module (and weight)

    Args:
        inputs: the input tensor to be normalized
        normalized_shape: the normalized_shape for normalization
        in_place: if to perform the operations in-place

    Returns:
        normalized tensor

    """
    if normalized_shape is None:
        normalized_shape = inputs.shape[-1]
    if isinstance(normalized_shape, numbers.Integral):
        normalized_shape = (normalized_shape,)

    if in_place:
        # This seems to create small discrepancy in result
        dim = list(range(len(inputs.shape))[-len(normalized_shape):])
        inputs -= inputs.mean(axis=dim, keepdim=True)
        inputs *= tinygrad.Tensor.rsqrt(inputs.var(axis=dim, keepdim=True) + 1e-5)
        return inputs
    else:
        # The commented out implementation also works, but uses the LayerNorm class and has the normalized_shape argument.
        #layernorm = LayerNorm(normalized_shape=normalized_shape, eps=1e-5, elementwise_affine=False)
        #return layernorm(inputs)
        return tinygrad.Tensor.layernorm(inputs, eps=1e-5)
        


def recursive_to(obj: typing.Any, **kwargs) -> typing.Any:
    r"""
    Just to move things to space
    *args is removed because it brings problems in using .cpu()

    Args:
        obj (): the object to move
        kwargs (): different keyword arguments

    Returns:
        cuda tensors in its original construct

    """
    if isinstance(obj, tinygrad.Tensor):
        try:
            return obj.to(**kwargs)
        except RuntimeError:
            kwargs.pop("non_blocking")
            return obj.to(**kwargs)
    elif isinstance(obj, list):
        return [recursive_to(o, **kwargs) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to(o, **kwargs) for o in obj)
    elif isinstance(obj, set):
        return set(recursive_to(o, **kwargs) for o in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to(v, **kwargs) for k, v in obj.items()}
    elif hasattr(obj, "to"):
        # this takes care of classes that implements the ~to method
        return obj.to(**kwargs)
    else:
        return obj


# =============================================================================
# Classes
# =============================================================================
# =============================================================================
# Tests
# =============================================================================
if __name__ == "__main__":
    pass
