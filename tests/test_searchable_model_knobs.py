import unittest

import torch
from omegaconf import OmegaConf

from module import ARPredictor, prediction_loss
from train import prediction_loss_kwargs


class SearchableModelKnobsTest(unittest.TestCase):
    def test_prediction_loss_matches_mse_without_detach(self):
        pred = torch.tensor([[1.0, 3.0]], requires_grad=True)
        target = torch.tensor([[0.0, 1.0]], requires_grad=True)

        loss = prediction_loss(pred, target, loss_type="mse", target_detach=False)
        loss.backward()

        self.assertTrue(torch.isclose(loss, torch.tensor(2.5)))
        self.assertIsNotNone(target.grad)

    def test_prediction_loss_can_detach_target(self):
        pred = torch.tensor([[1.0, 3.0]], requires_grad=True)
        target = torch.tensor([[0.0, 1.0]], requires_grad=True)

        loss = prediction_loss(pred, target, loss_type="mse", target_detach=True)
        loss.backward()

        self.assertTrue(torch.isclose(loss, torch.tensor(2.5)))
        self.assertIsNone(target.grad)

    def test_prediction_loss_supports_smooth_l1(self):
        pred = torch.tensor([[1.0, 3.0]])
        target = torch.tensor([[0.0, 1.0]])

        loss = prediction_loss(
            pred,
            target,
            loss_type="smooth_l1",
            smooth_l1_beta=1.0,
        )

        self.assertTrue(torch.isclose(loss, torch.tensor(1.0)))

    def test_prediction_loss_supports_mse_cosine(self):
        pred = torch.tensor([[1.0, 0.0]])
        target = torch.tensor([[0.0, 1.0]])

        loss = prediction_loss(
            pred,
            target,
            loss_type="mse_cosine",
            mse_weight=1.0,
            cosine_weight=0.1,
        )

        self.assertTrue(torch.isclose(loss, torch.tensor(1.1)))

    def test_prediction_loss_mse_cosine_can_detach_target(self):
        pred = torch.tensor([[1.0, 0.0]], requires_grad=True)
        target = torch.tensor([[0.0, 1.0]], requires_grad=True)

        loss = prediction_loss(
            pred,
            target,
            loss_type="mse_cosine",
            target_detach=True,
            mse_weight=1.0,
            cosine_weight=0.1,
        )
        loss.backward()

        self.assertIsNone(target.grad)

    def test_prediction_loss_kwargs_reads_cfg_fields(self):
        cfg = OmegaConf.create(
            {
                "loss": {
                    "pred": {
                        "type": "mse_cosine",
                        "target_detach": True,
                        "smooth_l1_beta": 0.5,
                        "mse_weight": 1.25,
                        "cosine_weight": 0.2,
                    }
                }
            }
        )

        kwargs = prediction_loss_kwargs(cfg)

        self.assertEqual(
            kwargs,
            {
                "loss_type": "mse_cosine",
                "target_detach": True,
                "smooth_l1_beta": 0.5,
                "mse_weight": 1.25,
                "cosine_weight": 0.2,
            },
        )

    def test_arpredictor_supports_add_conditioning(self):
        predictor = ARPredictor(
            num_frames=3,
            depth=1,
            heads=2,
            mlp_dim=16,
            input_dim=8,
            hidden_dim=8,
            output_dim=8,
            dim_head=4,
            conditioning_type="add",
        )
        x = torch.randn(2, 3, 8)
        c = torch.randn(2, 3, 8)

        out = predictor(x, c)

        self.assertEqual(out.shape, x.shape)

    def test_arpredictor_rejects_unknown_conditioning(self):
        with self.assertRaisesRegex(ValueError, "Unknown conditioning_type"):
            ARPredictor(
                num_frames=3,
                depth=1,
                heads=2,
                mlp_dim=16,
                input_dim=8,
                hidden_dim=8,
                output_dim=8,
                dim_head=4,
                conditioning_type="bad-mode",
            )


if __name__ == "__main__":
    unittest.main()
