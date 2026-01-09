"""
Per-Expert Gradient Decomposition by Modality

Tracks how much gradient each expert receives from RGB vs Action tokens.

核心原理：
    MoE输出: output = Σ (gate_weight_i * expert_i(input))
    反向传播: grad_expert_i = grad_output * gate_weight_i

    通过在backward hook中读取gate权重，可以将梯度分解为各模态的贡献。

安全保证：
    - 只读取梯度，不修改
    - 只存储标量值(.item())，不保留计算图
    - 不返回任何值，梯度原样传递
"""

import threading
from typing import Dict, Optional
import torch


class ExpertGradTracker:
    """
    全局单例：追踪每个专家收到的RGB/Action梯度贡献

    线程安全：使用threading.Lock保证多线程环境下的数据一致性
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # 存储结构: {layer_idx: {expert_idx: {'rgb': float, 'action': float, 'total': float}}}
        self._grad_data: Dict[int, Dict[int, Dict[str, float]]] = {}
        self._enabled = False
        self._modality_names = {0: "rgb", 1: "action", 2: "depth", 3: "force"}
        self._initialized = True

    def enable(self):
        """启用梯度追踪"""
        self._enabled = True
        self._grad_data.clear()  # 清空之前的数据

    def disable(self):
        """禁用梯度追踪"""
        self._enabled = False

    def is_enabled(self) -> bool:
        """是否启用"""
        return self._enabled

    def record_routing_info(self, layer_idx: int, routing_info: dict):
        """
        记录forward时的路由信息（由SparseMoeBlock调用）

        Args:
            layer_idx: MoE层的索引
            routing_info: {
                'modality_ids': Tensor[N],       # 每个token的模态ID
                'topk_idx': Tensor[N, top_k],     # 每个token选择的专家索引
                'topk_weight': Tensor[N, top_k],  # 对应的gate权重
            }
        """
        if not self._enabled:
            return

        # 存储路由信息供backward hook使用（detach避免内存泄漏）
        self._routing_info = {
            'modality_ids': routing_info['modality_ids'].detach().clone() if routing_info['modality_ids'] is not None else None,
            'topk_idx': routing_info['topk_idx'].detach().clone(),
            'topk_weight': routing_info['topk_weight'].detach().clone(),
        }

    def record_expert_gradient(
        self,
        layer_idx: int,
        expert_idx: int,
        grad_output: torch.Tensor,
        expert_rank: Optional[int] = None
    ):
        """
        在backward hook中调用，分解梯度并记录

        Args:
            layer_idx: MoE层索引
            expert_idx: 专家索引
            grad_output: 该专家输出的梯度 [N_tokens, hidden_dim]
            expert_rank: 该专家在top_k中的排名（0=第1个，1=第2个）
        """
        if not self._enabled or not hasattr(self, '_routing_info'):
            return

        info = self._routing_info
        if info is None or info['modality_ids'] is None:
            return

        modality_ids = info['modality_ids']          # [N]
        topk_idx = info['topk_idx']                  # [N, top_k]
        topk_weight = info['topk_weight']            # [N, top_k]

        # grad_output形状: [N, hidden_dim] (经过repeat_interleave后)
        # 需要还原到原始token顺序
        n_original = modality_ids.shape[0]
        top_k = topk_idx.shape[1]
        hidden_dim = grad_output.shape[-1]

        # grad是repeat_interleave后的，需要分组
        # 原始每个token对应top_k个expert，现在被展平了
        grad = grad_output.view(n_original, top_k, hidden_dim)  # [N, top_k, D]

        # 初始化该专家的梯度记录
        if layer_idx not in self._grad_data:
            self._grad_data[layer_idx] = {}
        if expert_idx not in self._grad_data[layer_idx]:
            self._grad_data[layer_idx][expert_idx] = {'rgb': 0.0, 'action': 0.0, 'total': 0.0}

        # 对每个模态分别计算梯度贡献
        for mod_id, mod_name in [(0, 'rgb'), (1, 'action'), (2, 'depth'), (3, 'force')]:
            # 找出该模态的token
            mod_mask = (modality_ids == mod_id)
            if not mod_mask.any():
                continue

            # 找出这些token是否选择了当前专家（在任一rank中）
            mod_topk_idx = topk_idx[mod_mask]       # [num_mod_tokens, top_k]
            mod_topk_weight = topk_weight[mod_mask] # [num_mod_tokens, top_k]
            mod_grad = grad[mod_mask]               # [num_mod_tokens, top_k, D]

            # 检查每个rank中是否有token选择了当前专家
            for rank in range(top_k):
                # 该模态的token中，把当前专家排在第rank位的
                rank_mask = (mod_topk_idx[:, rank] == expert_idx)
                if not rank_mask.any():
                    continue

                # 获取这些token的梯度和权重
                rank_grad = mod_grad[rank_mask, rank, :]  # [num_selected, D]
                rank_weight = mod_topk_weight[rank_mask, rank]  # [num_selected]

                # 加权梯度范数: ||grad * weight||
                # weight是标量，grad是向量，element-wise乘法
                weighted_grad = rank_grad * rank_weight.unsqueeze(-1)  # [num_selected, D]
                grad_norm = torch.norm(weighted_grad).item()  # 标量

                # 累加（取对数空间避免数值问题，或者直接累加）
                self._grad_data[layer_idx][expert_idx][mod_name] += grad_norm

        # 计算总梯度（所有模态的总和）
        total = sum(self._grad_data[layer_idx][expert_idx].values())
        self._grad_data[layer_idx][expert_idx]['total'] = total

    def get_gradient_summary(self) -> Dict:
        """
        获取梯度统计摘要

        Returns:
            {
                'layer_{idx}/expert_{idx}/rgb': float,
                'layer_{idx}/expert_{idx}/action': float,
                'layer_{idx}/expert_{idx}/total': float,
                ...
            }
        """
        summary = {}
        for layer_idx, experts in self._grad_data.items():
            for expert_idx, grad_dict in experts.items():
                prefix = f"layer_{layer_idx}/expert_{expert_idx}"
                for mod_name, value in grad_dict.items():
                    summary[f"grad_by_modality/{prefix}/{mod_name}"] = value

                # 计算action占比
                total = grad_dict.get('total', 1e-8)
                if total > 1e-8:
                    action_ratio = grad_dict.get('action', 0.0) / total
                    summary[f"grad_by_modality/{prefix}/action_ratio"] = action_ratio

        return summary

    def get_layer_aggregated(self) -> Dict:
        """
        获取聚合到层的统计（所有MoE层的总和）

        Returns:
            {
                'expert_0/rgb': float,
                'expert_0/action': float,
                ...
            }
        """
        aggregated = {}
        num_experts = 0

        # 聚合所有层
        for layer_idx, experts in self._grad_data.items():
            for expert_idx, grad_dict in experts.items():
                if expert_idx not in aggregated:
                    aggregated[expert_idx] = {'rgb': 0.0, 'action': 0.0, 'total': 0.0}
                for mod_name, value in grad_dict.items():
                    aggregated[expert_idx][mod_name] += value
            num_experts = max(num_experts, max(experts.keys()) + 1 if experts else 0)

        # 转换为返回格式
        result = {}
        for expert_idx, grad_dict in aggregated.items():
            for mod_name, value in grad_dict.items():
                result[f"expert_{expert_idx}/{mod_name}"] = value

        return result

    def clear(self):
        """清空当前记录的数据"""
        self._grad_data.clear()
        if hasattr(self, '_routing_info'):
            self._routing_info = None


# 全局实例
_tracker = ExpertGradTracker()


def enable_grad_tracking():
    """启用梯度追踪"""
    _tracker.enable()


def disable_grad_tracking():
    """禁用梯度追踪"""
    _tracker.disable()


def is_grad_tracking_enabled() -> bool:
    """检查是否启用梯度追踪"""
    return _tracker.is_enabled()


def record_routing_info(layer_idx: int, routing_info: dict):
    """记录路由信息"""
    _tracker.record_routing_info(layer_idx, routing_info)


def record_expert_gradient(layer_idx: int, expert_idx: int, grad_output: torch.Tensor):
    """记录专家梯度"""
    _tracker.record_expert_gradient(layer_idx, expert_idx, grad_output)


def get_gradient_summary() -> Dict:
    """获取梯度摘要"""
    return _tracker.get_gradient_summary()


def get_layer_aggregated() -> Dict:
    """获取层聚合统计"""
    return _tracker.get_layer_aggregated()


def clear_gradients():
    """清空梯度数据"""
    _tracker.clear()


def create_expert_backward_hook(layer_idx: int, expert_idx: int):
    """
    创建专家的backward hook

    这个hook会在专家FFN层的反向传播时被调用，
    用于分解和记录RGB/Action的梯度贡献。
    """
    def hook(module, grad_input, grad_output):
        """
        Args:
            module: MoeMLP实例
            grad_input: (grad_of_input, grad_of_fc1_weight, grad_of_fc1_bias, grad_of_fc2_weight, grad_of_fc2_bias)
            grad_output: (grad_of_output,)  shape: [N_tokens, intermediate_dim]
        """
        if not is_grad_tracking_enabled():
            return

        # grad_output[0]是该专家输出的梯度
        if grad_output[0] is not None:
            record_expert_gradient(layer_idx, expert_idx, grad_output[0])

        # 不返回任何值 = 不修改梯度，只是观察
        return

    return hook
