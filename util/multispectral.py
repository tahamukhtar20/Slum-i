"""Multispectral (Sentinel-2) input for ImageNet / DINOv2 pretrained backbones.

widen_first_conv replaces the backbone's first 3-channel convolution with an in_chans one. The
pretrained RGB filters go to the red/green/blue bands; every other band starts from the mean RGB
filter; all weights are scaled by 3/in_chans so the first activations keep their magnitude.
"""
import torch
from torch import nn

S2_RGB = (2, 1, 0)  # positions of red, green, blue in [blue, green, red, rededge1-3, nir, nir08, swir16, swir22]


def widen_first_conv(model, in_chans, rgb=S2_RGB):
    if in_chans == 3:
        return model
    for parent in model.modules():
        for name, conv in parent.named_children():
            if isinstance(conv, nn.Conv2d) and conv.in_channels == 3:
                new = nn.Conv2d(in_chans, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                                conv.dilation, conv.groups, conv.bias is not None).to(conv.weight.device)
                with torch.no_grad():
                    w = conv.weight
                    nw = w.mean(1, keepdim=True).repeat(1, in_chans, 1, 1)
                    for src, dst in enumerate(rgb):
                        nw[:, dst] = w[:, src]
                    new.weight.copy_(nw * 3.0 / in_chans)
                    if conv.bias is not None:
                        new.bias.copy_(conv.bias)
                setattr(parent, name, new)
                return model
    raise ValueError("no 3-channel Conv2d found to widen")
