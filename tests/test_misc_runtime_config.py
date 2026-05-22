import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils import misc


class TestMiscRuntimeConfig(unittest.TestCase):
    def test_define_gpus_respects_explicit_override(self):
        with mock.patch.dict(os.environ, {"SAM_CALIBRATION_CUDA_VISIBLE_DEVICES": "2,3"}, clear=False):
            misc.define_gpus("cifar10")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2,3")

    def test_update_dataroot_prefers_global_override(self):
        with mock.patch.dict(os.environ, {"SAM_CALIBRATION_DATA_ROOT": "/tmp/shared-data"}, clear=False):
            self.assertEqual(
                misc.update_dataroot("cifar10", "../data"),
                "/tmp/shared-data",
            )

    def test_update_dataroot_uses_imagenet_override_for_default_path(self):
        with mock.patch.dict(os.environ, {"IMAGENET_DATA_ROOT": "/datasets/imagenet"}, clear=False):
            self.assertEqual(
                misc.update_dataroot("imagenet", "../data"),
                "/datasets/imagenet",
            )

    def test_update_dataroot_keeps_explicit_existing_path(self):
        with tempfile.TemporaryDirectory() as data_root:
            with mock.patch.dict(os.environ, {"IMAGENET_DATA_ROOT": "/datasets/imagenet"}, clear=False):
                self.assertEqual(
                    misc.update_dataroot("imagenet", data_root),
                    data_root,
                )

    def test_update_dataroot_keeps_explicit_nondefault_path(self):
        custom_root = "/tmp/custom-imagenet-root"
        with mock.patch.dict(os.environ, {"IMAGENET_DATA_ROOT": "/datasets/imagenet"}, clear=False):
            self.assertEqual(
                misc.update_dataroot("imagenet", custom_root),
                custom_root,
            )

    def test_get_local_scratch_path_uses_first_available_env(self):
        with mock.patch.dict(os.environ, {"LOCAL_SCRATCH_DIR": "/scratch/local"}, clear=True):
            self.assertEqual(misc.get_local_scratch_path(), "/scratch/local")

    def test_get_local_scratch_path_raises_without_configuration(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                misc.get_local_scratch_path()


if __name__ == "__main__":
    unittest.main()
