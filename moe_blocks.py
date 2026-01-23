"""Sparse MoE building blocks ported from DiT-MoE for PAD."""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

GRAD_TRACKER_AVAILABLE = False
def record_routing_info(*args, **kwargs):
    pass
def is_grad_tracking_enabled():
    return False

def _approx_gelu():
    return nn.GELU(approximate="tanh")


class MoEGate(nn.Module):
    """Token-level gating network that selects top-k experts per token."""

    def __init__(
        self,
        embed_dim: int,
        num_experts: int = 16,
        num_experts_per_tok: int = 2,
        aux_loss_alpha: float = 0.01,
        use_modality_bias: bool = False,
        num_modalities: int = 3,
        modality_bias_init: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = num_experts
        self.scoring_func = "softmax"
        self.alpha = aux_loss_alpha
        self.seq_aux = False
        self.norm_topk_prob = True
        self.gating_dim = embed_dim
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.use_modality_bias = use_modality_bias
        self.num_modalities = num_modalities
        if self.use_modality_bias:
            bias = torch.zeros(self.num_modalities, self.n_routed_experts)
            if modality_bias_init is not None:
                modality_bias_init = modality_bias_init.detach().float()
                if modality_bias_init.shape == bias.shape:
                    bias.copy_(modality_bias_init)
            self.modality_bias = nn.Parameter(bias)
        else:
            self.modality_bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init = torch.nn.init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor, modality_ids: Optional[torch.Tensor] = None):
        """Compute gate indices, weights, and auxiliary balancing loss."""
        bsz, seq_len, hidden_dim = hidden_states.shape
        flat_states = hidden_states.reshape(-1, hidden_dim)
        logits = F.linear(flat_states, self.weight, None)
        flat_modality: Optional[torch.Tensor] = None
        if modality_ids is not None:
            flat_modality = modality_ids.reshape(-1)
        if self.use_modality_bias and flat_modality is not None and self.modality_bias is not None:
            if flat_modality.numel() == flat_states.shape[0]:
                bias = self.modality_bias.to(hidden_states.device)
                logits = logits + bias[flat_modality]

        scores = None
        if self.scoring_func == "softmax":
            if (not self.training) and self.norm_topk_prob:
                topk_logits, topk_idx = torch.topk(logits, k=self.top_k, dim=-1, sorted=False)
                topk_weight = topk_logits.softmax(dim=-1)
            else:
                scores = logits.softmax(dim=-1)
                topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
                if self.top_k > 1 and self.norm_topk_prob:
                    denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
                    topk_weight = topk_weight / denominator
        else:
            raise NotImplementedError(f"Unsupported MoE scoring function: {self.scoring_func}")

        aux_loss: Optional[torch.Tensor] = None
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            topk_idx_for_aux_loss = topk_idx.view(-1, self.top_k)

            if not self.seq_aux and flat_modality is not None:
                keep_mask = flat_modality != 1
                if keep_mask.any():
                    scores_for_aux = scores_for_aux[keep_mask]
                    topk_idx_for_aux_loss = topk_idx_for_aux_loss[keep_mask]
                else:
                    scores_for_aux = None

            if scores_for_aux is not None:
                if self.seq_aux:
                    scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                    ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                    ce.scatter_add_(
                        1,
                        topk_idx.view(bsz, -1),
                        torch.ones(bsz, seq_len * self.top_k, device=hidden_states.device),
                    ).div_(seq_len * self.top_k / self.n_routed_experts)
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
    """Single expert FFN using GELU+bias (aligns with DenseGeluMLP for init compatibility)."""

    def __init__(self, hidden_size: int, intermediate_size: int, pretraining_tp: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.pretraining_tp = pretraining_tp
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.act = _approx_gelu()
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DenseGeluMLP(nn.Module):
    """Dense FFN using GELU (matches the original DiT FFN)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.act = _approx_gelu()
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


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
        use_modality_bias: bool = False,
        num_modalities: int = 3,
        modality_bias_init: Optional[torch.Tensor] = None,
        collect_stats: bool = False,
        layer_idx: int = -1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.collect_stats = False
        self.layer_idx = layer_idx
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
            use_modality_bias=use_modality_bias,
            num_modalities=num_modalities,
            modality_bias_init=modality_bias_init,
        )
        self.n_shared_experts = n_shared_experts
        if self.n_shared_experts:
            shared_intermediate = embed_dim * self.n_shared_experts
            self.shared_experts = DenseGeluMLP(
                hidden_size=embed_dim,
                intermediate_size=shared_intermediate,
            )
        else:
            self.shared_experts = None
        self.last_aux_loss: Optional[torch.Tensor] = None
        self.last_routing_stats: Optional[dict] = None

    def forward(self, hidden_states: torch.Tensor, modality_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states, modality_ids=modality_ids)

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

        self.last_routing_stats = None
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
