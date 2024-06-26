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

"""
# =============================================================================
# Imports
# =============================================================================
import argparse
import numbers
import typing

import tinygrad.device
import tinygrad.tensor
from tinygrad import nn as tnn
import torch
from torch import nn
import tinygrad
import numpy as np

from omegafold import utils
from omegafold.utils.conversion import Module, Sequential, to_tinygrad, to_torch, dt2tg, dv2trch, dv2tg, dt2trch


# =============================================================================
# Constants
# =============================================================================
# =============================================================================
# Functions
# =============================================================================
def softmax(
        x: torch.Tensor,
        dim: int,
        *,
        dtype: typing.Optional[torch.dtype] = None,
        in_place: bool = False
) -> torch.Tensor:
    """
    In-place or normal softmax

    Args:
        x: the input tensor
        dim: the dimension along which to perform the softmax
        dtype: the data type
        in_place: if to perform inplace

    Returns:

    """
    x = to_tinygrad(x, checknan=False)

    ### In place removed because tinygrad's kernel fusion makes it redundant
    return to_torch(x.softmax(axis=dim), device=dv2trch[x.device])

# Define einsum formulae for different shapes
einsum_formulae = {
    (2, 2, 2): ("id, jd -> ij", "ij, jd -> id"),
    (3, 3, 3): ("bid, bjd -> bij", "bij, bjd -> bid"),
    (5, 5, 5): ("abcid, abcjd -> abcij", "abcij, abcjd -> abcid")
}

def _attention(
        query: tinygrad.Tensor,
        key: tinygrad.Tensor,
        scale: tinygrad.Tensor,
        value: tinygrad.Tensor,
        bias: tinygrad.Tensor,
        return_edge: bool,
        edge_reduction: str,
        edge_reduction_dim: int
) -> typing.Tuple[tinygrad.Tensor, typing.Optional[tinygrad.Tensor]]:
    """Normal attention

    Args:
        query: positive tensor of shape (*_q, dim_qk)
        key: positive tensor of shape (*_k, dim_qk)
        scale: the scaling of logits
        value: tensor of shape (*_k, dim_v)
        bias: the bias acting as either mask or relative positional encoding
        return_edge: if to return the logits of attention

    Returns:
        The aggregated tensor of shape (*_q, dim_v)

    """
    q_shape = query.shape
    k_shape = key.shape
    v_shape = value.shape

    logits_formula, out_formula = einsum_formulae[(len(q_shape), len(k_shape), len(v_shape))]
    
    logits = tinygrad.Tensor.einsum(logits_formula, query * scale, key)
    logits = logits + bias
    attn = softmax(logits, dim=-1, in_place=not return_edge)
    attn = to_tinygrad(attn)
    out = tinygrad.Tensor.einsum(out_formula, attn, value)
    if return_edge:
        attn = getattr(attn, edge_reduction)(axis=edge_reduction_dim)
        return out, attn
    else:
        return out, None


def attention(
        query: torch.Tensor,
        key: torch.Tensor,
        scale: typing.Union[torch.Tensor, float],
        value: torch.Tensor,
        bias: torch.Tensor,
        subbatch_size: typing.Optional[int] = None,
        *,
        return_edge: bool = False,
        edge_reduction: str = 'sum',
        edge_reduction_dim: int = 0,
        tt = True
) -> typing.Tuple[torch.Tensor, typing.Tuple[torch.Tensor]]:
    """Computes attention with q, k , v

    Args:
        query: positive tensor of shape (*_q, dim_qk)
        key: positive tensor of shape (*_k, dim_qk)
        scale: the scaling of logits
        value: tensor of shape (*_k, dim_v)
        bias: the bias acting as either mask or relative positional encoding
        subbatch_size: the subbatch size to split the computation into
        return_edge: if to return the logits
        edge_reduction:
        edge_reduction_dim:

    Returns:
        The aggregated tensor of shape (*_q, dim_v)

    """
    query = to_tinygrad(query)
    scale = to_tinygrad(scale)
    key = to_tinygrad(key)
    value = to_tinygrad(value)
    bias = to_tinygrad(bias)
    q_length, k_length, v_dim = query.shape[-2], key.shape[-2], value.shape[-1]
    subbatch_size = subbatch_size or q_length

    batch_shape = list(query.shape[:-2])

    output = tinygrad.Tensor.empty(*batch_shape, q_length, v_dim, device=query.device, dtype=query.dtype)

    if return_edge:
        batch_shape.pop(edge_reduction_dim + 2)
        attns = tinygrad.Tensor.empty(
            *batch_shape, q_length, k_length, device=query.device, dtype=query.dtype
        )
    else:
        attns = None

    for i, q_i in enumerate(query.split(subbatch_size, dim=-2)):
        start, end = i * subbatch_size, (i + 1) * subbatch_size,
        if bias.shape[-2] != q_length:
            b_i = bias
        else:
            b_i = bias[..., start:end, :]

        res, attn = _attention(
            q_i, key, scale, value, b_i, return_edge,
            edge_reduction, edge_reduction_dim
        )

        output[..., start:end, :] = res
        if return_edge:
            attns[..., start:end, :] = attn


    return (to_torch(output), to_torch(attns)) if tt else (output, attns)


# =============================================================================
# Classes
# =============================================================================

class OFModule(Module):
    """
    The OmegaFold modules
        args: The arguments used for each of the modules
    """

    def __init__(
            self,
            cfg: typing.Optional[argparse.Namespace]
    ) -> None:
        #super(OFModule, self).__init__()
        self.cfg = cfg

    @property
    def device(self) -> torch.device:
        return self.getdevice()

    @property
    def dtype(self) -> torch.dtype:
        return self.getdtype()


activations = {
    "relu": tinygrad.Tensor.relu,
    "leaky_relu": tinygrad.Tensor.leakyrelu,
    "sigmoid": tinygrad.Tensor.sigmoid,
    "tanh": tinygrad.Tensor.tanh,
}
class Transition(OFModule):
    def __init__(self, d: int, n: int, activation: str) -> None:
        super(Transition, self).__init__(None)
        fc1 = tnn.Linear(d, n * d)
        fc2 = tnn.Linear(n * d, d)
        try:
            act = activations[activation.lower()]
        except:
            raise ValueError(f"Activation {activation} not supported")
        self.network = Sequential(fc1, act, fc2)

    def forward(
            self,
            x: torch.Tensor,
            subbatch_size: typing.Optional[int]
    ) -> torch.Tensor:
        x = to_tinygrad(x)
        subbatch_size = subbatch_size or x.shape[-2]

        out = tinygrad.Tensor.empty(x.shape, device=x.device, dtype=x.dtype)
        for i, x_i in enumerate(x.split(subbatch_size, dim=0)):
            start, end = i * subbatch_size, (i + 1) * subbatch_size
            x_i = utils.tnormalize(x_i)
            out[start:end] = self.network(x_i)
        return to_torch(out)


class MultiHeadedScaling(OFModule):
    """
    Perform an element wise scale shift

    """

    def __init__(
            self,
            shape: typing.Union[int, typing.List[int], torch.Size],
            num_heads: int,
            on_out_ready: typing.Optional[
                typing.Callable[[torch.Tensor], torch.Tensor]
            ],
            dtype: typing.Optional[torch.dtype] = None,
    ) -> None:
        """

        Args:
            shape: the shape of the input dimensions
            num_heads: the number of dimensions to squeeze to
            dtype: the dtype of the parameters at generation
            on_out_ready: the function called on exit
        """
        super(MultiHeadedScaling, self).__init__(None)
        factory_kwargs = nn.factory_kwargs({"dtype": dtype})
        if isinstance(shape, numbers.Integral):
            shape = (shape,)
        shape = list(tuple(shape))
        self.unsqueeze_dim = - (len(shape) + 1)
        shape.insert(0, num_heads)
        self.shape = shape
        self.split_dims = [1] * num_heads
        self.weight = tinygrad.Tensor.empty(self.shape, dtype=dt2tg[dtype])
        self.bias = tinygrad.Tensor.empty(self.shape, dtype=dt2tg[dtype])
        self.call_on_out_ready = on_out_ready

        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> typing.List[tinygrad.Tensor]:
        """
        Element wise multiplication followed by addition

        Args:
            x: the input tensor with the trailing dimensions following
                ~self.shape

        Returns:
            A output tensor of the same shape

        """
        x = to_tinygrad(x)
        x = x.unsqueeze(self.unsqueeze_dim) * self.weight + self.bias
        positive_index = x.ndim + self.unsqueeze_dim
        if self.call_on_out_ready is not None:
            x = self.call_on_out_ready(x)

        x = x.split(self.split_dims, dim=positive_index)

        return [x_i.squeeze(positive_index) for x_i in x]

    def reset_parameters(self):
        self.weight = tinygrad.Tensor.normal(self.weight.shape, std=0.02, dtype=self.weight.dtype)
        self.bias = tinygrad.Tensor.zeros(self.bias.shape, dtype=self.bias.dtype)


class Val2ContBins(OFModule):
    def __init__(self, cfg: argparse.Namespace, ):
        super(Val2ContBins, self).__init__(cfg)

        x_bin_size = (cfg.x_max - cfg.x_min) / (cfg.x_bins - 2)

        self.x_offset = tinygrad.Tensor(np.linspace(
            cfg.x_min - x_bin_size / 2,
            cfg.x_max + x_bin_size / 2,
            cfg.x_bins
        ))
        self.no_load("x_offset")
        self.coeff : float = -0.5 / ((x_bin_size * 0.2) ** 2)
        self.no_load("coeff")
        # `*0.5`: makes it not too blurred

    def forward(self, dist_x):  # (*)
        dist_x = to_tinygrad(dist_x)
        x_offset_shape = [1] * len(dist_x.size()) + [len(self.x_offset)]
        x = dist_x.unsqueeze(-1) - self.x_offset.view(*x_offset_shape)
        x_norm = self.coeff * x.pow(2)
        x_norm = x_norm - x_norm.max(-1, keepdim=True)[0]
        logits = x_norm.softmax(-1)

        return to_torch(logits)


class Val2Bins(OFModule):
    """
    Convert continuous values to bins

    Attributes:
        breaks: the line space break
    """

    def __init__(self, cfg: argparse.Namespace) -> None:
        super(Val2Bins, self).__init__(cfg)
        self.breaks = tinygrad.Tensor(np.linspace(
                cfg.first_break, cfg.last_break, cfg.num_bins - 1
        ))
        self.no_load("breaks")

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """

        Args:
            dist: distances in the euclidean space.

        Returns:

        """
        dist : tinygrad.Tensor = to_tinygrad(dist)
        dist = dist.unsqueeze(-1)
        dist_bin : tinygrad.Tensor = (dist > self.breaks).cast(tinygrad.dtypes.int32)
        dist_bin = dist_bin.sum(-1)
        return to_torch(dist_bin)


class Node2Edge(OFModule):
    """Communicate between tracks

        faster than OutProductMean mostly due to a better implementation
    """

    def __init__(self, in_dim: int, proj_dim: int, out_dim: int) -> None:
        super(Node2Edge, self).__init__(None)
        self.input_proj = tnn.Linear(in_dim, proj_dim * 2)
        self.proj_dim = proj_dim
        self.out_weights = tinygrad.Tensor.empty(proj_dim, proj_dim, out_dim)
        self.out_bias = tinygrad.Tensor.empty(out_dim)

    def forward(
            self, node_repr: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        mask = to_tinygrad(mask)
        node_repr = to_tinygrad(node_repr)
        
        node_repr = utils.tnormalize(node_repr)

        act = self.input_proj(node_repr)
        mask = mask[..., None]
        act = act * mask

        # This einsum may be a pain to convert to tinygrad
        norm = tinygrad.Tensor.einsum("sid, sjd->ijd", mask, mask)


        l, r = act.split(self.proj_dim, dim=-1)
        # We found this implementation to work significantly faster
        out = tinygrad.Tensor.einsum(
            'sid, def, sje-> ijf', l, self.out_weights, r
        ) + self.out_bias
        out = out / (norm + 1e-3)

        return to_torch(out)


class Attention(OFModule):
    """
    Widely used attention mechanism

    Attributes:
        qg_weights (nn.Parameter): weight matrices for queries and gates
        qg_bias (nn.Parameter): biases for queries and gates
        kv_weights (nn.Parameter): weight matrices for queries and gates
        kv_bias (nn.Linear): biases for keys and values

        o_weights (nn.Linear): the output weight matrix
        o_bias (nn.Linear): the output bias
    """

    def __init__(
            self,
            q_dim: int,
            kv_dim: int,
            n_head: int,
            gating: bool,
            c: int,
            out_dim: int,
            n_axis: int
    ) -> None:
        super(Attention, self).__init__(None)
        self.c = c
        self.n_head = n_head
        self.gating = gating
        self.q_dim = q_dim
        self.n_axis = n_axis

        self.qg_weights = tinygrad.Tensor.empty(q_dim, n_axis, n_head, (gating + 1) * c)
        
        self.kv_weights = tinygrad.Tensor.empty(kv_dim, n_axis, n_head, 2 * c)
        
        self.qg_bias = tinygrad.Tensor.empty(n_axis, n_head, 1, c * (1 + gating))
        
        self.kv_bias = tinygrad.Tensor.empty(n_axis, n_head, 1, c * 2)

        self.o_weights = tinygrad.Tensor.empty(n_axis, n_head, c, out_dim)
        self.o_bias = tinygrad.Tensor.empty([out_dim, n_axis])

    def forward(
            self,
            q_inputs: torch.Tensor,
            kv_inputs: torch.Tensor,
            bias: torch.Tensor,
            *,
            fwd_cfg: typing.Optional[argparse.Namespace] = None,
            tt = True
    ) -> typing.Union[typing.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Perform the standard multi-headed attention with added gating with some
        biases

        Args:
            q_inputs: the inputs to generate query vectors,
                of shape (*, q_len, q_dim, (n_axis))
            kv_inputs: the inputs to generate key and value vectors,
                of shape (*, kv_len, kv_dim, (n_axis))
            bias: the bias for the logits
                of shape (*, n_head, q_len, kv_len)
            fwd_cfg: if return logits

        Return:
            output tensor (*, seq_len, o_dim, (n_axis))
            attention logits (Optional) (q_len, kv_len, num_head)
        """
        q_inputs = to_tinygrad(q_inputs)
        kv_inputs = to_tinygrad(kv_inputs)
        bias = to_tinygrad(bias)

        # Acquire the q, k, v tensors
        to_unsqueeze = (
                q_inputs.shape[-1] != self.n_axis and
                q_inputs.shape[-1] == self.q_dim
        )
        if to_unsqueeze:
            q_inputs = q_inputs.unsqueeze(-1)
            kv_inputs = kv_inputs.unsqueeze(-1)
            if bias is not None:
                bias = bias.unsqueeze(-4)

        attn_out = self._get_attn_out(q_inputs, kv_inputs, fwd_cfg, bias)

        output = tinygrad.Tensor.einsum('brhqc,rhco->bqor', attn_out, self.o_weights)
        output += self.o_bias

        if to_unsqueeze:
            output = output.squeeze(-1)
        return to_torch(output) if tt else output

    def _get_attn_out(self, q_inputs, kv_inputs, fwd_cfg, bias):

        qg = tinygrad.Tensor.einsum("bqar,arhc->brhqc", q_inputs, self.qg_weights)
        qg += self.qg_bias
        q_out = qg.split(self.c, dim=-1)
        q = q_out[0]

        kv = tinygrad.Tensor.einsum("bkar,arhc->brhkc", kv_inputs, self.kv_weights)
        kv += self.kv_bias
        k, v = kv.split([self.c, self.c], dim=-1)

        # Attention
        subbatch_size = (
            q.shape[-4] if fwd_cfg is None else fwd_cfg.subbatch_size
        )
        attn_out, _ = attention(
            query=q,
            key=k,
            value=v,
            subbatch_size=subbatch_size,
            bias=bias,
            scale=self.c ** (-0.5),
            tt=False
        )
        # get the gating
        if self.gating:
            g = q_out[1].sigmoid()
            attn_out *= g

        return attn_out


class AttentionWEdgeBias(OFModule):
    def __init__(
            self,
            d_node: int,
            d_edge: int,
            n_head: int,
            attn_gating: bool,
            attn_c: int
    ) -> None:
        super(AttentionWEdgeBias, self).__init__(None)
        self.proj_edge_bias = tnn.Linear(
            in_features=d_edge, out_features=n_head  # , bias=False
        )
        self.attention = Attention(
            q_dim=d_node,
            kv_dim=d_node,
            n_head=n_head,
            gating=attn_gating,
            c=attn_c,
            out_dim=d_node,
            n_axis=1
        )

    def forward(
            self,
            node_repr: torch.Tensor,
            edge_repr: torch.Tensor,
            mask: torch.Tensor,
            *,
            fwd_cfg: typing.Optional[argparse.Namespace] = None
    ) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, torch.Tensor]]:
        """

        Args:
            node_repr:
            edge_repr:
            mask:
            fwd_cfg:

        Returns:

        """
        node_repr = to_tinygrad(node_repr)
        edge_repr = to_tinygrad(edge_repr)
        mask = to_tinygrad(mask)
        node_repr = utils.tnormalize(node_repr)
        edge_repr = utils.tnormalize(edge_repr)
        # check dim
        edge_bias = self.proj_edge_bias(edge_repr).permute(2, 0, 1)

        edge_bias = edge_bias + utils.tmask2bias(mask[..., None, None, :])
        attn_out = self.attention(
            node_repr, node_repr, bias=edge_bias, fwd_cfg=fwd_cfg
        )
        return to_torch(attn_out)


def _get_sharded_stacked(
        edge_repr: tinygrad.Tensor,
        subbatch_size: int
):
    subbatch_size = subbatch_size or edge_repr.shape[-2]
    idx = 0
    start, end = 0, subbatch_size
    while start < edge_repr.shape[-2]:
        yield start, end, tinygrad.Tensor.stack(
            [
                edge_repr[start:end],
                edge_repr.transpose(-2, -3)[start:end]
            ], dim=-1
        )
        idx += 1
        start, end = idx * subbatch_size, (idx + 1) * subbatch_size

def tg_glu(inp : tinygrad.Tensor) -> tinygrad.Tensor:
    # Implements a * b.sigmoid() where a and b are the two halves of the input
    a, b = inp.split(inp.shape[-1] // 2, dim=-1)
    return a * b.sigmoid()

class GeometricAttention(OFModule):
    """We have a lot of stuff here for GRAM reduction

    """

    def __init__(self, d_edge: int, c: int, n_head: int, n_axis: int) -> None:
        super(GeometricAttention, self).__init__(None)
        self.d_edge = d_edge
        self.n_axis = n_axis
        self.n_head = n_head
        self.linear_b_weights = tinygrad.Tensor.empty([d_edge, n_axis, n_head])
        
        self.linear_b_bias = tinygrad.Tensor.empty([n_axis, n_head, 1, 1])
        

        self.act_w = tinygrad.Tensor.empty([d_edge, n_axis, d_edge * 5])
        
        self.act_b = tinygrad.Tensor.empty([n_axis, d_edge * 5])

        self.out_proj_w = tinygrad.Tensor.empty([n_axis, d_edge, d_edge])
        
        self.out_proj_b = tinygrad.Tensor.empty([n_axis, d_edge])
        #self.glu = nn.GLU()

        self.attention = Attention(
            q_dim=d_edge,
            kv_dim=d_edge,
            n_head=n_head,
            c=c,
            gating=True,
            out_dim=d_edge,
            n_axis=n_axis
        )

    def _get_attended(
            self,
            edge_repr: tinygrad.Tensor,
            mask: tinygrad.Tensor,
            fwd_cfg
    ) -> torch.Tensor:
        attended = tinygrad.Tensor.empty(
            *edge_repr.shape, self.n_axis,
            dtype=edge_repr.dtype,
            device=edge_repr.device
        )

        b = tinygrad.Tensor.ones(
            self.n_axis, self.n_head, *edge_repr.shape[:2],
            dtype=edge_repr.dtype,
            device=edge_repr.device
        )
        '''
        This is objectively very stupid. For some reason, b += mask2bias is not working. I replaced it with a multiplication followed by subtraction, because with
        mask values near infinity, 0 + mask ~= mask - 1 and 0 + 0 = 1 - 1

        This is dumb, but it works and I'm tired of debugging this
        '''
        b = (b * utils.tmask2bias(mask)) - 1    # HACK: see above

        for s, e, edge_r in _get_sharded_stacked(
                edge_repr, subbatch_size=fwd_cfg.subbatch_size
        ):
            b[..., s:e, :] = tinygrad.Tensor.einsum(
                'qkcr,crh->rhqk', edge_r, self.linear_b_weights
            ) + self.linear_b_bias
        for s, e, edge_r in _get_sharded_stacked(
                edge_repr, subbatch_size=fwd_cfg.subbatch_size
        ):
            attended[s:e] = self.attention(
                edge_r, edge_r, b, fwd_cfg=fwd_cfg, tt=False
            )
        attended = attended
        return attended[..., 0] + attended[..., 1].transpose(-2, -3)

    def _get_gated(self, edge_repr: tinygrad.Tensor, mask: tinygrad.Tensor, fwd_cfg):
        gated = tinygrad.Tensor.empty(
            *edge_repr.shape[:2],
            self.n_axis,
            self.d_edge,
            device=edge_repr.device,
            dtype=edge_repr.dtype
        )
        for s_row, e_row, edge_row in _get_sharded_stacked(
                edge_repr, subbatch_size=fwd_cfg.subbatch_size
        ):
            act_row = self._get_act_row(edge_row, mask[s_row:e_row])
            act_g = (
                tinygrad.Tensor.einsum(
                    'abdr,drc->abrc',
                    edge_row,
                    self.act_w[..., -self.d_edge:]
                ) + self.act_b[..., -self.d_edge:]).sigmoid()
            
            for s_col, e_col, edge_col, in _get_sharded_stacked(
                    edge_repr, subbatch_size=fwd_cfg.subbatch_size
            ):
                act_col = self._get_act_col(edge_col, mask[s_col:e_col])
                ab = tinygrad.Tensor.einsum('ikrd,jkrd->ijrd', act_row, act_col)
                ab = utils.tnormalize(ab.contiguous())
                ab = to_tinygrad(ab)
                gated[s_row:e_row, s_col:e_col] = tinygrad.Tensor.einsum(
                    'abrd,rdc->abrc', ab, self.out_proj_w
                )
                gated[s_row:e_row, s_col:e_col].add(self.out_proj_b)
                gated[s_row:e_row, s_col:e_col] *= act_g[:, s_col:e_col]

        return gated.sum(-2)

    def _get_sliced_weight(self, weight: tinygrad.Tensor, shift=0) -> tinygrad.Tensor:
        w = weight[..., :-self.d_edge].unflatten(-1, sizes=(4, -1))
        w = w[..., shift::2, :]
        w = w.flatten(start_dim=-2)
        return w

    def _get_act_row(
            self,
            edge_row: tinygrad.Tensor,
            mask: tinygrad.Tensor
    ) -> tinygrad.Tensor:
        w = self._get_sliced_weight(self.act_w)
        b = self._get_sliced_weight(self.act_b)
        act = tinygrad.Tensor.einsum('abdr,drc->abrc', edge_row, w) + b
        act = tg_glu(act) * mask[..., None, None, None]
        return act

    def _get_act_col(
            self,
            edge_row: tinygrad.Tensor,
            mask: tinygrad.Tensor
    ) -> tinygrad.Tensor:
        w = self._get_sliced_weight(self.act_w, shift=1)
        b = self._get_sliced_weight(self.act_b, shift=1)
        act = tinygrad.Tensor.einsum('abdr,drc->abrc', edge_row, w) + b
        act = tg_glu(act) * mask[..., None, None, None]
        return act

    def forward(
            self, edge_repr: torch.Tensor, mask: torch.Tensor, fwd_cfg
    ) -> torch.Tensor:
        edge_repr = utils.tnormalize(edge_repr)
        edge_repr = to_tinygrad(edge_repr)
        mask = to_tinygrad(mask)
        out = self._get_attended(edge_repr, mask, fwd_cfg)
        out += self._get_gated(edge_repr, mask, fwd_cfg)

        return to_torch(out)


# =============================================================================
# Tests
# =============================================================================
if __name__ == '__main__':
    pass
