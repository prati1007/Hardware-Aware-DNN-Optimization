import torch
import torch.nn as nn

def prune_conv_layer(layer, prune_ratio=0.3):
    weights = layer.weight.data.abs().mean(dim=(1,2,3))
    num_prune = int(prune_ratio * weights.shape[0])

    prune_idx = torch.argsort(weights)[:num_prune]
    keep_idx = torch.argsort(weights)[num_prune:]

    new_layer = nn.Conv2d(
        in_channels=layer.in_channels,
        out_channels=len(keep_idx),
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding
    )

    new_layer.weight.data = layer.weight.data[keep_idx]
    return new_layer
