# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Optional
import torch

__all__ = [
    'MLP',
    'GLU',
]

class MLP(torch.nn.Module):

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        moe_num_experts: int,
        activation_fn: Callable,
        device: Optional[torch.device],
    ) -> None:
        super().__init__()

        self.moe_num_experts: int = moe_num_experts
        self.ffn_hidden_size: int = ffn_hidden_size
        self.hidden_size: int = hidden_size
        self.activation_fn: Callable = activation_fn

        self.w1 = torch.nn.Parameter(
            torch.rand(
                moe_num_experts * ffn_hidden_size,
                hidden_size,
                device=device,
            ),
        )
        self.w2 = torch.nn.Parameter(
            torch.rand(
                moe_num_experts * ffn_hidden_size,
                hidden_size,
                device=device,
            ),
        )
        self.activation_fn = activation_fn

    def forward(self, x: torch.Tensor, expert_idx: int) -> torch.Tensor:
        expert_w1 = self.w1.view(
            self.moe_num_experts,
            self.ffn_hidden_size,
            self.hidden_size,
        )[expert_idx]
        expert_w2 = self.w2.view(
            self.moe_num_experts,
            self.ffn_hidden_size,
            self.hidden_size,
        )[expert_idx]

        before_activation = x @ expert_w1.t()
        layer_1_output = self.activation_fn(before_activation)
        output = layer_1_output @ expert_w2
        return output


class GLU(torch.nn.Module):

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        moe_num_experts: int,
        activation_fn: Callable,
        device: Optional[torch.device],
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.moe_num_experts = moe_num_experts

        self.w1 = torch.nn.Parameter(
            torch.rand(
                moe_num_experts * ffn_hidden_size,
                hidden_size,
                device=device,
            ),
        )
        self.v1 = torch.nn.Parameter(
            torch.rand(
                moe_num_experts * ffn_hidden_size,
                hidden_size,
                device=device,
            ),
        )
        self.w2 = torch.nn.Parameter(
            torch.rand(
                moe_num_experts * ffn_hidden_size,
                hidden_size,
                device=device,
            ),
        )
        self.activation_fn = activation_fn

    def forward(self, x: torch.Tensor, expert_idx: torch.Tensor):
        expert_w1 = self.w1.view(
            self.moe_num_experts,
            self.ffn_hidden_size,
            self.hidden_size,
        )[expert_idx]
        expert_v1 = self.v1.view(
            self.moe_num_experts,
            self.ffn_hidden_size,
            self.hidden_size,
        )[expert_idx]
        expert_w2 = self.w2.view(
            self.moe_num_experts,
            self.ffn_hidden_size,
            self.hidden_size,
        )[expert_idx]

        x1 = x.matmul(expert_w1.t())
        x2 = x.matmul(expert_v1.t())
        x1 = self.activation_fn(x1)
        x1 = x1 * x2
        x1 = x1.matmul(expert_w2)
        return x1