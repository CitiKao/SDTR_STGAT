import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stgat_model import STGATPredictor  # noqa: E402


class STGATSensorVModeTest(unittest.TestCase):
    def test_forward_v_node_mode_outputs_node_horizons(self) -> None:
        edge_index = torch.tensor(
            [
                [0, 1],
                [1, 0],
                [1, 2],
                [2, 1],
            ],
            dtype=torch.long,
        )
        edge_lengths = torch.ones(edge_index.shape[0], dtype=torch.float32)
        adjacency = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        model = STGATPredictor(
            num_nodes=3,
            edge_index=edge_index,
            edge_lengths=edge_lengths,
            adj_matrix=adjacency,
            hidden_dim=8,
            num_heads=2,
            num_st_blocks=1,
            num_gtcn_layers=1,
            pred_horizon=4,
            adaptive_topk=1,
            speed_use_adaptive=True,
            v_domain="node",
        )

        speed_seq = torch.rand(2, 3, 12, dtype=torch.float32)
        temporal_context = torch.rand(2, 12, 6, dtype=torch.float32)
        output = model.forward_v(speed_seq, temporal_context)

        self.assertEqual(tuple(output.shape), (2, 3, 4))
        summary = model.speed_adaptive_graph_summary()
        self.assertIsNotNone(summary)
        self.assertEqual(summary["domain"], "node")

    def test_forward_v_edge_mode_outputs_edge_horizons(self) -> None:
        edge_index = torch.tensor(
            [
                [0, 1],
                [1, 2],
                [2, 0],
            ],
            dtype=torch.long,
        )
        edge_lengths = torch.ones(edge_index.shape[0], dtype=torch.float32)
        adjacency = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        model = STGATPredictor(
            num_nodes=3,
            edge_index=edge_index,
            edge_lengths=edge_lengths,
            adj_matrix=adjacency,
            hidden_dim=8,
            num_heads=2,
            num_st_blocks=1,
            num_gtcn_layers=1,
            pred_horizon=4,
            adaptive_topk=1,
            speed_use_adaptive=True,
            speed_adaptive_topk=1,
            v_domain="edge",
        )

        speed_seq = torch.rand(2, 3, 12, dtype=torch.float32)
        temporal_context = torch.rand(2, 12, 6, dtype=torch.float32)
        output = model.forward_v(speed_seq, temporal_context)

        self.assertEqual(tuple(output.shape), (2, 3, 4))
        summary = model.speed_adaptive_graph_summary()
        self.assertIsNotNone(summary)
        self.assertEqual(summary["domain"], "edge")

    def test_node_speed_adaptive_graph_uses_speed_specific_topk(self) -> None:
        edge_index = torch.tensor(
            [
                [0, 1],
                [1, 0],
                [1, 2],
                [2, 1],
                [2, 3],
                [3, 2],
            ],
            dtype=torch.long,
        )
        edge_lengths = torch.ones(edge_index.shape[0], dtype=torch.float32)
        adjacency = torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        model = STGATPredictor(
            num_nodes=4,
            edge_index=edge_index,
            edge_lengths=edge_lengths,
            adj_matrix=adjacency,
            hidden_dim=8,
            num_heads=2,
            num_st_blocks=1,
            num_gtcn_layers=1,
            pred_horizon=4,
            adaptive_topk=2,
            speed_use_adaptive=True,
            speed_adaptive_topk=1,
            v_domain="node",
        )

        summary = model.speed_adaptive_graph_summary()
        self.assertIsNotNone(summary)
        self.assertEqual(summary["domain"], "node")
        self.assertEqual(summary["degree_min"], 1)
        self.assertLessEqual(summary["degree_max"], 2)
        self.assertLessEqual(summary["num_edges"], 8)
        self.assertEqual(summary["self_loops_kept"], 4)

    def test_weighted_fixed_graph_and_history_mask_flow_into_node_v_mode(self) -> None:
        edge_index = torch.tensor(
            [
                [0, 1],
                [1, 0],
                [1, 2],
                [2, 1],
            ],
            dtype=torch.long,
        )
        edge_lengths = torch.tensor([1.2, 1.1, 0.8, 0.9], dtype=torch.float32)
        adjacency = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        adjacency_weights = torch.tensor(
            [
                [0.0, 0.7, 0.0],
                [0.2, 0.0, 0.5],
                [0.0, 0.4, 0.0],
            ],
            dtype=torch.float32,
        )
        model = STGATPredictor(
            num_nodes=3,
            edge_index=edge_index,
            edge_lengths=edge_lengths,
            adj_matrix=adjacency,
            adj_weight_matrix=adjacency_weights,
            hidden_dim=8,
            num_heads=2,
            num_st_blocks=1,
            num_gtcn_layers=1,
            pred_horizon=4,
            adaptive_topk=1,
            speed_use_adaptive=True,
            use_speed_history_mask=True,
            v_domain="node",
        )

        fixed_weight_map = {
            (int(recv), int(send)): float(weight)
            for (recv, send), weight in zip(
                model.fixed_edge_index.T.tolist(),
                model.fixed_edge_weights.tolist(),
            )
        }
        self.assertAlmostEqual(fixed_weight_map[(0, 1)], 0.7, places=6)
        self.assertAlmostEqual(fixed_weight_map[(1, 0)], 0.2, places=6)
        self.assertAlmostEqual(fixed_weight_map[(1, 2)], 0.5, places=6)
        self.assertAlmostEqual(fixed_weight_map[(0, 0)], 1.0, places=6)

        speed_seq = torch.rand(2, 3, 12, dtype=torch.float32)
        temporal_context = torch.rand(2, 12, 6, dtype=torch.float32)
        speed_history_mask = torch.randint(0, 2, (2, 3, 12), dtype=torch.bool)
        output = model.forward_v(
            speed_seq,
            temporal_context,
            speed_history_mask_seq=speed_history_mask,
        )

        self.assertEqual(tuple(output.shape), (2, 3, 4))

    def test_node_speed_mode_can_disable_fixed_edge_length_feature(self) -> None:
        edge_index = torch.tensor(
            [
                [0, 1],
                [1, 0],
                [1, 2],
                [2, 1],
            ],
            dtype=torch.long,
        )
        edge_lengths = torch.tensor([1.2, 1.1, 0.8, 0.9], dtype=torch.float32)
        adjacency = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        model = STGATPredictor(
            num_nodes=3,
            edge_index=edge_index,
            edge_lengths=edge_lengths,
            adj_matrix=adjacency,
            hidden_dim=8,
            num_heads=2,
            num_st_blocks=1,
            num_gtcn_layers=1,
            pred_horizon=4,
            adaptive_topk=1,
            speed_use_adaptive=True,
            use_fixed_edge_length_feature=False,
            v_domain="node",
        )

        speed_seq = torch.rand(2, 3, 12, dtype=torch.float32)
        temporal_context = torch.rand(2, 12, 6, dtype=torch.float32)
        output = model.forward_v(speed_seq, temporal_context)

        self.assertEqual(tuple(output.shape), (2, 3, 4))
        self.assertFalse(model.use_fixed_edge_length_feature)
        self.assertEqual(model.fixed_edge_feat_dim, 1)

    def test_edge_speed_mode_includes_edge_length_input_channel_when_enabled(self) -> None:
        edge_index = torch.tensor(
            [
                [0, 1],
                [1, 2],
                [2, 0],
            ],
            dtype=torch.long,
        )
        edge_lengths = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float32)
        adjacency = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        model = STGATPredictor(
            num_nodes=3,
            edge_index=edge_index,
            edge_lengths=edge_lengths,
            adj_matrix=adjacency,
            hidden_dim=8,
            num_heads=2,
            num_st_blocks=1,
            num_gtcn_layers=1,
            pred_horizon=2,
            adaptive_topk=1,
            speed_use_adaptive=False,
            use_fixed_edge_length_feature=True,
            v_domain="edge",
        )

        speed_seq = torch.zeros(1, 3, 2, dtype=torch.float32)
        edge_input = model._build_edge_input(speed_seq)

        self.assertTrue(model.use_edge_domain_length_feature)
        self.assertEqual(model.edge_input_dim, 2)
        self.assertEqual(tuple(edge_input.shape), (1, 3, 2, 2))
        self.assertTrue(torch.allclose(edge_input[0, :, :, 1], edge_lengths[:, None]))


if __name__ == "__main__":
    unittest.main()
