import unittest

import torch

from deepjscc.metrics import per_image_psnr
from deepjscc.model import DeepJSCC, PowerNormalization, WirelessChannel, bandwidth_ratio_to_channels


class ModelTest(unittest.TestCase):
    def test_paper_bandwidth_ratios(self) -> None:
        self.assertEqual(bandwidth_ratio_to_channels(1 / 12), 8)
        self.assertEqual(bandwidth_ratio_to_channels(1 / 6), 16)

    def test_power_normalization_is_per_image(self) -> None:
        z = torch.randn(4, 8, 8, 8)
        normalized = PowerNormalization()(z)
        k = normalized[0].numel() / 2
        energy = normalized.square().flatten(1).sum(dim=1)
        self.assertTrue(torch.allclose(energy, torch.full_like(energy, k), atol=1e-4))

    def test_awgn_component_variance(self) -> None:
        torch.manual_seed(0)
        z = torch.zeros(1024, 8, 8, 8)
        noisy = WirelessChannel("awgn")(z, snr_db=0)
        self.assertTrue(torch.isclose(noisy.var(), torch.tensor(0.5), atol=0.02))

    def test_model_shape_and_backward(self) -> None:
        model = DeepJSCC(1 / 12, "rayleigh")
        images = torch.rand(2, 3, 32, 32)
        reconstruction = model(images, 7)
        self.assertEqual(reconstruction.shape, images.shape)
        self.assertTrue(torch.isfinite(per_image_psnr(images, reconstruction)).all())
        reconstruction.mean().backward()


if __name__ == "__main__":
    unittest.main()
