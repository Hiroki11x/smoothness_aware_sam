import math
import pathlib
import sys
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt import OptimizerSetting, build_optimizer, canonical_optimizer_name, requires_previous_model
from opt.adsgd import Adsgd
from opt.sam import SAM


class TestSASAMPaperBehavior(unittest.TestCase):
    def test_canonical_optimizer_name(self):
        cases = [
            ("adsgd", "sasgd"),
            ("sasgd", "sasgd"),
            ("adsgd_sam", "sasam"),
            ("sasam", "sasam"),
            ("sam", "sam"),
        ]

        for name, expected in cases:
            with self.subTest(name=name):
                self.assertEqual(canonical_optimizer_name(name), expected)

    def test_requires_previous_model(self):
        cases = [
            ("adsgd", True),
            ("sasgd", True),
            ("adsgd_sam", True),
            ("sasam", True),
            ("sam", False),
            ("adam", False),
        ]

        for name, expected in cases:
            with self.subTest(name=name):
                self.assertIs(requires_previous_model(name), expected)

    def test_build_optimizer_accepts_paper_name_for_sasam(self):
        model = torch.nn.Linear(2, 1)

        optimizer = build_optimizer(
            OptimizerSetting(
                name="sasam",
                lr=0.09,
                weight_decay=0.0,
                model=model,
            )
        )

        self.assertIsInstance(optimizer, SAM)
        self.assertTrue(optimizer.param_groups[0]["adaptive_rho"])

    def test_legacy_adsgd_sam_alias_maps_to_paper_sasam_behavior(self):
        model = torch.nn.Linear(2, 1)

        optimizer = build_optimizer(
            OptimizerSetting(
                name="adsgd_sam",
                lr=0.09,
                weight_decay=0.0,
                model=model,
            )
        )

        self.assertIsInstance(optimizer, SAM)
        self.assertTrue(optimizer.param_groups[0]["adaptive_rho"])

    def test_sasam_updates_rho_from_current_learning_rate(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = SAM(
            [parameter],
            base_optimizer=Adsgd,
            rho=0.05,
            eps=1e-12,
            adaptive_rho=True,
            lr=0.09,
            amplifier=0.02,
        )

        parameter.grad = torch.tensor([1.0])
        optimizer.first_step(zero_grad=False)

        self.assertAlmostEqual(optimizer.param_groups[0]["rho"], math.sqrt(0.09))

    def test_vanilla_sam_keeps_fixed_rho(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = SAM(
            [parameter],
            base_optimizer=torch.optim.SGD,
            rho=0.05,
            eps=1e-12,
            adaptive_rho=False,
            lr=0.09,
        )

        parameter.grad = torch.tensor([1.0])
        optimizer.first_step(zero_grad=False)

        self.assertAlmostEqual(optimizer.param_groups[0]["rho"], 0.05)


if __name__ == "__main__":
    unittest.main()
