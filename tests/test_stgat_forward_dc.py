import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stgat_model import STGATPredictor


def build_tiny_model() -> STGATPredictor:
    torch.manual_seed(7)
    edge_index = torch.tensor(
        [
            [0, 1],
            [1, 2],
            [2, 0],
        ],
        dtype=torch.long,
    )
    edge_lengths = torch.ones(edge_index.shape[0], dtype=torch.float32)
    adj = torch.zeros((3, 3), dtype=torch.float32)
    adj[edge_index[:, 0], edge_index[:, 1]] = 1.0
    return STGATPredictor(
        num_nodes=3,
        edge_index=edge_index,
        edge_lengths=edge_lengths,
        adj_matrix=adj,
        hidden_dim=4,
        num_heads=1,
        num_st_blocks=1,
        num_gtcn_layers=1,
        kernel_size=3,
        pred_horizon=2,
        node_feat_dim=2,
        adaptive_topk=2,
    ).eval()


class STGATForwardDCTest(unittest.TestCase):
    def test_forward_dc_matches_full_forward_dc_heads(self) -> None:
        model = build_tiny_model()
        node_seq = torch.randn(2, 3, 4, 2)
        speed_seq = torch.randn(2, 3, 4)

        with torch.inference_mode():
            d_full, c_full, _ = model(node_seq, speed_seq)
            d_dc, c_dc = model.forward_dc(node_seq, speed_seq)

        self.assertTrue(torch.allclose(d_dc, d_full, atol=1e-6))
        self.assertTrue(torch.allclose(c_dc, c_full, atol=1e-6))

    def test_forward_dc_does_not_run_speed_prediction_path(self) -> None:
        model = build_tiny_model()
        node_seq = torch.randn(1, 3, 4, 2)
        speed_seq = torch.randn(1, 3, 4)

        def forbidden_speed_path(*args: object, **kwargs: object) -> torch.Tensor:
            raise AssertionError("forward_dc should not run the V prediction path.")

        model._run_speed_path = forbidden_speed_path  # type: ignore[method-assign]

        with torch.inference_mode():
            d_dc, c_dc = model.forward_dc(node_seq, speed_seq)

        self.assertEqual(tuple(d_dc.shape), (1, 3, 2))
        self.assertEqual(tuple(c_dc.shape), (1, 3, 2))


if __name__ == "__main__":
    unittest.main()
