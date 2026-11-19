"""Smoke tests for the notebook-derived CBAM-RSU2Net architecture."""

import unittest

try:
    import torch
    from colorectal_seg.cbam_rsu2net import CBAM_RSU2Net
except ImportError:
    torch = None
    CBAM_RSU2Net = None


@unittest.skipIf(torch is None, "PyTorch is not installed in this lightweight test environment.")
class TestCBAMRSU2Net(unittest.TestCase):
    def test_forward_shape_and_parameter_count(self):
        torch.set_num_threads(2)
        model = CBAM_RSU2Net(in_ch=1, out_ch=3).eval()
        with torch.no_grad():
            output = model(torch.randn(1, 1, 16, 16, 16))

        self.assertEqual(tuple(output.shape), (1, 3, 16, 16, 16))
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            6_209_705,
        )

    def test_rejects_mismatched_class_count(self):
        with self.assertRaises(ValueError):
            CBAM_RSU2Net(in_ch=1, out_ch=2)


if __name__ == "__main__":
    unittest.main()
