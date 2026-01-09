"""Sparse MoE building blocks ported from DiT-MoE for PAD."""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

# Import gradient tracking utilities
try:
    from grad_tracker import record_routing_info, is_grad_tracking_enabled
    GRAD_TRACKER_AVAILABLE = True
except ImportError:
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
        # Normalize the top-k probabilities so they sum to 1 after truncation.
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
        self.last_raw_scores = None  # Store raw logits for debugging/analysis

    def reset_parameters(self) -> None:
        init = torch.nn.init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor, modality_ids: Optional[torch.Tensor] = None):
        """Compute gate indices, weights, and auxiliary balancing loss."""
        bsz, seq_len, hidden_dim = hidden_states.shape
        flat_states = hidden_states.reshape(-1, hidden_dim)
        logits = F.linear(flat_states, self.weight, None)
        logits_before_bias = logits.clone() if self.training else None  # Save before bias for analysis
        flat_modality: Optional[torch.Tensor] = None
        if modality_ids is not None:
            flat_modality = modality_ids.reshape(-1)
        if self.use_modality_bias and flat_modality is not None and self.modality_bias is not None:
            if flat_modality.numel() == flat_states.shape[0]:
                bias = self.modality_bias.to(hidden_states.device)
                logits = logits + bias[flat_modality]

        # Store raw scores for analysis (detach to avoid memory leak)
        if self.training:
            with torch.no_grad():
                self.last_raw_scores = {
                    'logits': logits.detach().clone(),
                    'logits_before_bias': logits_before_bias.detach().clone() if logits_before_bias is not None else None,
                    'modality_ids': flat_modality.detach().clone() if flat_modality is not None else None,
                }
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

            # Exclude action tokens (modality_id == 1) from load-balancing aux loss so
            # action embeddings are not driven by router regularization.
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
        # Pretraining TP path removed to keep parity with DenseGeluMLP and simplify init sharing.
        return self.fc2(self.act(self.fc1(x)))


class DenseGeluMLP(nn.Module):
    """Dense FFN using GELU (matches the original DiT FFN)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        # Keep bias to align with the original dense MLP and avoid systematic shifts.
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
        layer_idx: int = -1,  # 添加层索引用于梯度追踪
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.collect_stats = collect_stats
        self.layer_idx = layer_idx  # 记录层索引
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
            # Use GELU dense FFN so shared path matches the original dense model.
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

        # Record routing info for gradient tracking (only during training and when enabled)
        if self.training and is_grad_tracking_enabled() and modality_ids is not None:
            flat_modality = modality_ids.reshape(-1)
            routing_info = {
                'modality_ids': flat_modality,
                'topk_idx': topk_idx.view(-1, self.num_experts_per_tok),
                'topk_weight': topk_weight.view(-1, self.num_experts_per_tok),
            }
            record_routing_info(self.layer_idx, routing_info)

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

        # All tokens (including action) now go through normal MoE routing
        if self.shared_experts is not None:
            output = output + self.shared_experts(identity)

        # lightweight routing stats for logging (detached, no grad)
        self.last_routing_stats = None
        if modality_ids is not None:
            with torch.no_grad():
                flat_mod = modality_ids.reshape(-1)
                flat_topk = topk_idx.view(-1, self.num_experts_per_tok)
                flat_weight = topk_weight.view(-1, self.num_experts_per_tok)
                num_experts = self.num_experts

                # Modality names for logging
                modality_names = {0: "rgb", 1: "action", 2: "depth", 3: "force"}

                # Basic stats (always collected for compatibility)
                stats = {}
                if flat_mod[flat_mod == 1].any():
                    a_topk = flat_topk[flat_mod == 1]
                    hit = (a_topk == 0).any(dim=1).float().mean()
                    counts = torch.bincount(a_topk.view(-1), minlength=num_experts)
                    coverage = (counts > 0).float().mean()
                    stats["action_hit_rate"] = hit
                    stats["action_coverage"] = coverage
                    a_weight = flat_weight[flat_mod == 1]
                    hit_weight = a_weight[a_topk == 0]
                    if hit_weight.numel() > 0:
                        stats["action_expert0_weight"] = hit_weight.mean()

                if flat_mod[flat_mod == 0].any():
                    rgb_topk = flat_topk[flat_mod == 0]
                    counts = torch.bincount(rgb_topk.view(-1), minlength=num_experts)
                    stats["rgb_coverage"] = (counts > 0).float().mean()

                if flat_mod[flat_mod == 2].any():
                    depth_topk = flat_topk[flat_mod == 2]
                    counts = torch.bincount(depth_topk.view(-1), minlength=num_experts)
                    stats["depth_coverage"] = (counts > 0).float().mean()

                # Detailed stats (only when collect_stats=True)
                if self.collect_stats:
                    # Get routing probabilities from gate (logits -> softmax)
                    gate_scores = self.gate.last_raw_scores
                    if gate_scores is not None and gate_scores.get('logits') is not None:
                        all_logits = gate_scores['logits']  # (total_tokens, num_experts)
                        all_probs = torch.softmax(all_logits, dim=-1)

                        # For each modality, compute detailed stats
                        for mod_id in sorted(set(flat_mod.cpu().tolist())):
                            mod_name = modality_names.get(mod_id, f"mod_{mod_id}")
                            mask = flat_mod == mod_id
                            if not mask.any():
                                continue

                            mod_probs = all_probs[mask]  # (num_mod_tokens, num_experts)
                            mod_topk = flat_topk[mask]     # (num_mod_tokens, top_k)
                            mod_weight = flat_weight[mask] # (num_mod_tokens, top_k)
                            num_mod_tokens = mask.sum().item()

                            # Token count
                            stats[f"{mod_name}/token_count"] = num_mod_tokens

                            # Top-1 expert histogram (count per expert)
                            top1_indices = mod_topk[:, 0]  # (num_mod_tokens,)
                            top1_hist = torch.bincount(top1_indices, minlength=num_experts).float()
                            stats[f"{mod_name}/top1_hist"] = top1_hist  # Will be normalized after all-reduce

                            # Top-k expert histogram (count per expert, expanded)
                            topk_expanded = mod_topk.view(-1)  # (num_mod_tokens * top_k,)
                            topk_hist = torch.bincount(topk_expanded, minlength=num_experts).float()
                            stats[f"{mod_name}/topk_hist"] = topk_hist  # Will be normalized after all-reduce

                            # Routing entropy (normalized by log(num_experts))
                            # H = -sum(p * log(p)), normalized = H / log(E)
                            entropy = -(mod_probs * torch.log(mod_probs + 1e-10)).sum(dim=-1)
                            normalized_entropy = entropy / torch.log(torch.tensor(num_experts, dtype=torch.float32))
                            stats[f"{mod_name}/entropy"] = normalized_entropy.mean()

                            # Router confidence: top1 probability mean and margin
                            top1_probs = mod_probs.max(dim=-1)[0]  # (num_mod_tokens,)
                            stats[f"{mod_name}/top1_prob_mean"] = top1_probs.mean()

                            # Margin = E[p_top1 - p_top2] (only meaningful if top_k >= 2)
                            if self.num_experts_per_tok >= 2:
                                sorted_probs, _ = torch.sort(mod_probs, dim=-1, descending=True)
                                margin = (sorted_probs[:, 0] - sorted_probs[:, 1]).mean()
                                stats[f"{mod_name}/margin_mean"] = margin
                            else:
                                # top_k=1时，margin无意义（只有一个选择），设为0或跳过
                                stats[f"{mod_name}/margin_mean"] = torch.tensor(0.0)

                if stats:
                    # ensure plain tensors detached (handle both tensor and scalar types)
                    self.last_routing_stats = {}
                    for k, v in stats.items():
                        if isinstance(v, torch.Tensor):
                            self.last_routing_stats[k] = v.detach()
                        else:
                            self.last_routing_stats[k] = v
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

    def get_gate_scores(self):
        """Get the last gate scores for analysis/debugging."""
        return getattr(self.gate, "last_raw_scores", None)
