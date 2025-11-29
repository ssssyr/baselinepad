"""Sparse MoE building blocks ported from DiT-MoE for PAD."""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class MoEGate(nn.Module):
    """Token-level gating network that selects top-k experts per token."""

    def __init__(
        self,
        embed_dim: int,
        num_experts: int = 16,
        num_experts_per_tok: int = 2,
        aux_loss_alpha: float = 0.01,
    ):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = num_experts
        self.scoring_func = "softmax"
        self.alpha = aux_loss_alpha
        self.seq_aux = False
        self.norm_topk_prob = False
        self.gating_dim = embed_dim
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init = torch.nn.init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor):
        """Compute gate indices, weights, and auxiliary balancing loss."""
        bsz, seq_len, hidden_dim = hidden_states.shape
        flat_states = hidden_states.reshape(-1, hidden_dim)
        logits = F.linear(flat_states, self.weight, None)
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f"Unsupported MoE scoring function: {self.scoring_func}")

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        aux_loss: Optional[torch.Tensor]
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1),
                    num_classes=self.n_routed_experts,
                )
                ce = mask_ce.float().mean(0)
                pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss


class AddAuxiliaryLoss(torch.autograd.Function):
    """Adds auxiliary loss contribution without modifying the main loss scalar."""

    @staticmethod
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor):
        if aux_loss is not None:
            assert aux_loss.numel() == 1
            ctx.dtype = aux_loss.dtype
            ctx.requires_aux = aux_loss.requires_grad
        else:
            ctx.dtype = output.dtype
            ctx.requires_aux = False
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_loss = None
        if ctx.requires_aux:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


class MoeMLP(nn.Module):
    """Single expert feed-forward network."""

    def __init__(self, hidden_size: int, intermediate_size: int, pretraining_tp: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.pretraining_tp = pretraining_tp
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pretraining_tp > 1:
            slice_size = self.intermediate_size // self.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice_size, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice_size, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice_size, dim=1)

            gate_proj = torch.cat([F.linear(x, gate) for gate in gate_proj_slices], dim=-1)
            up_proj = torch.cat([F.linear(x, up) for up in up_proj_slices], dim=-1)
            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice_size, dim=-1)
            down_proj = [F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.pretraining_tp)]
            return sum(down_proj)

        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class SparseMoeBlock(nn.Module):
    """Sparse mixture of MoeMLP experts with optional shared experts."""

    def __init__(
        self,
        embed_dim: int,
        mlp_ratio: float = 4.0,
        num_experts: int = 16,
        num_experts_per_tok: int = 2,
        pretraining_tp: int = 1,
        aux_loss_alpha: float = 0.01,
        n_shared_experts: int = 2,
    ):
        super().__init__()
        self.num_experts_per_tok = num_experts_per_tok
        intermediate_size = int(mlp_ratio * embed_dim)
        self.experts = nn.ModuleList(
            [
                MoeMLP(
                    hidden_size=embed_dim,
                    intermediate_size=intermediate_size,
                    pretraining_tp=pretraining_tp,
                )
                for _ in range(num_experts)
            ]
        )
        self.gate = MoEGate(
            embed_dim=embed_dim,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            aux_loss_alpha=aux_loss_alpha,
        )
        self.n_shared_experts = n_shared_experts
        if self.n_shared_experts:
            shared_intermediate = embed_dim * self.n_shared_experts
            self.shared_experts = MoeMLP(
                hidden_size=embed_dim,
                intermediate_size=shared_intermediate,
                pretraining_tp=pretraining_tp,
            )
        else:
            self.shared_experts = None
        self.last_aux_loss: Optional[torch.Tensor] = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)

        flat_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            flat_states = flat_states.repeat_interleave(self.num_experts_per_tok, dim=0)
            routed = torch.empty_like(flat_states, dtype=flat_states.dtype)
            for expert_idx, expert in enumerate(self.experts):
                mask = flat_topk_idx == expert_idx
                if mask.any():
                    routed[mask] = expert(flat_states[mask]).float()
            routed = (routed.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            output = routed.view(*orig_shape)
            if aux_loss is not None:
                self.last_aux_loss = aux_loss.detach()
                output = AddAuxiliaryLoss.apply(output, aux_loss)
            else:
                self.last_aux_loss = None
        else:
            self.last_aux_loss = None
            output = self.moe_infer(flat_states, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)

        if self.shared_experts is not None:
            output = output + self.shared_experts(identity)
        return output

    @torch.no_grad()
    def moe_infer(self, x: torch.Tensor, flat_expert_indices: torch.Tensor, flat_expert_weights: torch.Tensor):
        """Deterministic routing path for eval."""
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount(minlength=len(self.experts)).cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok
        for expert_idx, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if expert_idx == 0 else tokens_per_expert[expert_idx - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[expert_idx]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache = expert_cache.to(expert_out.dtype)
            expert_cache.scatter_reduce_(
                0,
                exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]),
                expert_out,
                reduce="sum",
            )
        return expert_cache

