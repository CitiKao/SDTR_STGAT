"""
replay_buffer.py — 經驗回放緩衝區

儲存 (state, action, reward, next_state, done, action_mask, next_action_mask)
並支援均勻隨機取樣供 Double DQN 訓練使用。
"""

from __future__ import annotations

import random
from collections import deque
from typing import List, NamedTuple

import numpy as np
import torch


class Transition(NamedTuple):
    state: np.ndarray            # (state_dim,)
    action: int
    reward: float
    next_state: np.ndarray       # (state_dim,)
    done: bool
    action_mask: np.ndarray      # (max_neighbors,)
    next_action_mask: np.ndarray # (max_neighbors,)


class BatchTensors(NamedTuple):
    """從 replay buffer 取樣後轉為 GPU-ready tensors"""
    states: torch.Tensor            # (B, state_dim)
    actions: torch.Tensor           # (B, 1)  long
    rewards: torch.Tensor           # (B, 1)
    next_states: torch.Tensor       # (B, state_dim)
    dones: torch.Tensor             # (B, 1)
    action_masks: torch.Tensor      # (B, max_neighbors)
    next_action_masks: torch.Tensor # (B, max_neighbors)


class ReplayBuffer:
    """固定容量的經驗回放緩衝區，使用 deque 自動淘汰最舊資料。"""

    def __init__(self, capacity: int = 50_000) -> None:
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        action_mask: np.ndarray,
        next_action_mask: np.ndarray,
    ) -> None:
        self.buffer.append(
            Transition(state, action, reward, next_state, done, action_mask, next_action_mask)
        )

    def sample(self, batch_size: int, device: torch.device) -> BatchTensors:
        batch: List[Transition] = random.sample(self.buffer, batch_size)

        states = np.stack([t.state for t in batch])
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch])
        dones = np.array([t.done for t in batch], dtype=np.float32)
        action_masks = np.stack([t.action_mask for t in batch])
        next_action_masks = np.stack([t.next_action_mask for t in batch])

        return BatchTensors(
            states=torch.tensor(states, dtype=torch.float32, device=device),
            actions=torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1),
            rewards=torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1),
            next_states=torch.tensor(next_states, dtype=torch.float32, device=device),
            dones=torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1),
            action_masks=torch.tensor(action_masks, dtype=torch.float32, device=device),
            next_action_masks=torch.tensor(next_action_masks, dtype=torch.float32, device=device),
        )

    def __len__(self) -> int:
        return len(self.buffer)
