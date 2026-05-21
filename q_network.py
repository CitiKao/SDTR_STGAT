"""
q_network.py — Q 網路（MLP + 節點 Embedding）

網路結構：
  1. 將 current_node 與 destination 各自通過 Embedding 層取得向量
  2. 與連續特徵（鄰居預測速度、鄰居邊長、時間槽）拼接
  3. 經過多層 MLP
  4. 輸出 max_neighbors 維 Q-values
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Parameters
    ----------
    num_nodes      : 圖中節點總數（Embedding 詞表大小）
    max_neighbors  : 最大鄰居數 = 動作空間大小
    embed_dim      : 節點 Embedding 維度
    hidden_dim     : 隱藏層寬度
    """

    def __init__(
        self,
        num_nodes: int,
        max_neighbors: int,
        embed_dim: int = 32,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.max_neighbors = max_neighbors
        self.embed = nn.Embedding(num_nodes, embed_dim)

        # 連續特徵維度：pred_speeds(M) + lengths(M) + time_slot(1)
        cont_dim = 2 * max_neighbors + 1
        input_dim = 2 * embed_dim + cont_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_neighbors),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state : (B, state_dim)  其中 state[:, 0] = current_node,
                state[:, 1] = destination, state[:, 2:] = 連續特徵

        Returns
        -------
        q_values : (B, max_neighbors)
        """
        current_node = state[:, 0].long()
        destination = state[:, 1].long()
        cont_features = state[:, 2:]

        curr_emb = self.embed(current_node)   # (B, embed_dim)
        dest_emb = self.embed(destination)     # (B, embed_dim)

        x = torch.cat([curr_emb, dest_emb, cont_features], dim=1)
        return self.net(x)
