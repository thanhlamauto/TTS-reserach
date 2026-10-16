from types import SimpleNamespace

import torch
import torch.nn.functional as F

from otlab.features import TensorFeatures


def test_dinov2_checkpoint_preprocessing_retains_gradient():
    encoder = object.__new__(TensorFeatures)
    encoder.size = (256, 256)
    encoder.processor = SimpleNamespace(do_center_crop=True, crop_size=dict(height=224, width=224))
    encoder.mean = torch.tensor([.485, .456, .406])[None, :, None, None]
    encoder.std = torch.tensor([.229, .224, .225])[None, :, None, None]
    x = torch.rand(1, 3, 512, 512, generator=torch.Generator().manual_seed(42), requires_grad=True)
    actual = encoder.preprocess(x)
    expected = (F.interpolate(x, (256, 256), mode='bicubic', align_corners=False, antialias=True)[:, :, 16:240, 16:240]
                - encoder.mean) / encoder.std
    assert torch.equal(actual, expected)
    assert actual.shape == (1, 3, 224, 224)
    assert actual.shape[-1] % 14 == 0
    grad = torch.autograd.grad(actual.square().mean(), x)[0]
    assert torch.isfinite(grad).all() and grad.norm() > 0
