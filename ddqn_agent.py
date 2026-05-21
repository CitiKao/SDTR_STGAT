"""
ddqn_agent.py — Double DQN Agent

核心演算法：
  y = r + γ · Q_target(s', argmax_a Q_online(s', a))

包含：
  - epsilon-greedy 動作選擇（含 action masking）
  - replay buffer 取樣訓練
  - target network 定期同步
  - 模型存取
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from q_network import QNetwork
from replay_buffer import ReplayBuffer


class DoubleDQNAgent:
    """
    Parameters
    ----------
    num_nodes       : 圖中節點總數
    max_neighbors   : 最大鄰居數 = 動作空間大小
    state_dim       : 狀態向量維度
    embed_dim       : 節點 Embedding 維度
    hidden_dim      : MLP 隱藏層寬度
    lr              : 學習率
    gamma           : 折扣因子
    epsilon_start   : 初始探索率
    epsilon_end     : 最低探索率
    epsilon_decay   : 線性衰減步數
    buffer_capacity : replay buffer 容量
    batch_size      : 訓練批次大小
    target_update   : 每隔幾次 learn() 同步 target network
    device          : 'cpu' 或 'cuda'
    """

    def __init__(
        self,
        num_nodes: int,
        max_neighbors: int,
        state_dim: int,
        *,
        embed_dim: int = 32,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 5000,
        buffer_capacity: int = 50_000,
        batch_size: int = 64,
        target_update: int = 200,
        device: str = "cpu",
    ) -> None:
        self.num_nodes = num_nodes
        self.max_neighbors = max_neighbors
        self.state_dim = state_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = torch.device(device)

        # ε-greedy 參數
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # 網路
        self.online_net = QNetwork(
            num_nodes, max_neighbors, embed_dim, hidden_dim
        ).to(self.device)
        self.target_net = copy.deepcopy(self.online_net)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_capacity)

        self.learn_step_count: int = 0

    # ── 動作選擇 ──────────────────────────────────────────────

    def select_action(
        self,
        state: np.ndarray,
        action_mask: np.ndarray,
        *,
        greedy: bool = False,
    ) -> int:
        """
        ε-greedy action selection with action masking.
        greedy=True 時使用純貪婪策略（評估用）。
        """
        valid_actions = np.where(action_mask > 0)[0]
        if len(valid_actions) == 0:
            return 0  # fallback，環境會判定非法

        if not greedy and np.random.random() < self.epsilon:
            return int(np.random.choice(valid_actions))

        with torch.no_grad():
            s = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            q = self.online_net(s).squeeze(0)
            # 非法動作的 Q-value 設為 -inf
            mask_t = torch.tensor(
                action_mask, dtype=torch.float32, device=self.device
            )
            q[mask_t == 0] = -float("inf")
            return int(q.argmax().item())

    # ── 訓練 ──────────────────────────────────────────────────

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        action_mask: np.ndarray,
        next_action_mask: np.ndarray,
    ) -> None:
        self.buffer.push(
            state, action, reward, next_state, done, action_mask, next_action_mask
        )

    def learn(self) -> Optional[float]:
        """
        從 replay buffer 取樣並執行一步 Double DQN 更新。
        回傳 loss 值；若 buffer 不夠大則回傳 None。
        """
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size, self.device)

        # ── online Q(s, a) ──
        q_online = self.online_net(batch.states)                    # (B, M)
        q_sa = q_online.gather(1, batch.actions)                    # (B, 1)

        # ── Double DQN target ──
        with torch.no_grad():
            has_next_action = batch.next_action_masks.sum(dim=1, keepdim=True) > 0
            # 用 online net 選動作
            q_next_online = self.online_net(batch.next_states)      # (B, M)
            # 非法動作 mask 為 -inf
            q_next_online[batch.next_action_masks == 0] = -float("inf")
            best_actions = q_next_online.argmax(dim=1, keepdim=True)  # (B, 1)

            # 用 target net 估計 Q 值
            q_next_target = self.target_net(batch.next_states)      # (B, M)
            q_next_val = q_next_target.gather(1, best_actions)      # (B, 1)
            q_next_val = torch.where(has_next_action, q_next_val, torch.zeros_like(q_next_val))

            target = batch.rewards + self.gamma * q_next_val * (1.0 - batch.dones)

        loss = self.loss_fn(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪防止爆炸
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # ε 衰減（線性）
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start
            - (self.epsilon_start - self.epsilon_end)
            * self.learn_step_count
            / self.epsilon_decay,
        )

        self.learn_step_count += 1

        # 定期同步 target network
        if self.learn_step_count % self.target_update == 0:
            self.sync_target()

        return loss.item()

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ── 模型持久化 ────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "online_state_dict": self.online_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "learn_step_count": self.learn_step_count,
                "num_nodes": self.num_nodes,
                "max_neighbors": self.max_neighbors,
                "state_dim": self.state_dim,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._validate_checkpoint_compatibility(ckpt)
        self.online_net.load_state_dict(ckpt["online_state_dict"])
        self.target_net.load_state_dict(ckpt["target_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.epsilon = ckpt["epsilon"]
        self.learn_step_count = ckpt["learn_step_count"]

    def _validate_checkpoint_compatibility(self, ckpt: dict) -> None:
        expected_num_nodes = int(
            ckpt.get(
                "num_nodes",
                ckpt["online_state_dict"]["embed.weight"].shape[0],
            )
        )
        expected_max_neighbors = int(
            ckpt.get(
                "max_neighbors",
                ckpt["online_state_dict"]["net.4.bias"].shape[0],
            )
        )
        expected_state_dim = int(
            ckpt.get(
                "state_dim",
                2 * expected_max_neighbors + 3,
            )
        )

        mismatches: list[str] = []
        if expected_num_nodes != self.num_nodes:
            mismatches.append(
                f"num_nodes checkpoint={expected_num_nodes}, current={self.num_nodes}"
            )
        if expected_max_neighbors != self.max_neighbors:
            mismatches.append(
                "max_neighbors "
                f"checkpoint={expected_max_neighbors}, current={self.max_neighbors}"
            )
        if expected_state_dim != self.state_dim:
            mismatches.append(
                f"state_dim checkpoint={expected_state_dim}, current={self.state_dim}"
            )

        if mismatches:
            mismatch_text = "; ".join(mismatches)
            raise ValueError(
                "Checkpoint is incompatible with the current DoubleDQNAgent "
                f"configuration: {mismatch_text}. "
                "Use a checkpoint trained on the same graph/action-space setup."
            )
