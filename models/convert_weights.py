"""Convert diffusers model weights to fit our model"""

from __future__ import annotations

from typing import Dict

import torch


def _map_vae_resblock(
    src: str,
    dst: str,
    diffusers_sd: Dict[str, torch.Tensor],
    new_sd: Dict[str, torch.Tensor],
) -> None:
    """Map a diffusers VAE ResnetBlock2D to our VAE ResidualBlock."""
    mapping = {
        'norm1': 'groupnorm1',
        'conv1': 'conv1',
        'norm2': 'groupnorm2',
        'conv2': 'conv2',
        'conv_shortcut': 'residual_conv',
    }
    for src_name, dst_name in mapping.items():
        for suffix in ('weight', 'bias'):
            key = f'{src}.{src_name}.{suffix}'
            if key in diffusers_sd:
                new_sd[f'{dst}.{dst_name}.{suffix}'] = diffusers_sd[key]


def _map_vae_attention(
    src: str,
    dst: str,
    diffusers_sd: Dict[str, torch.Tensor],
    new_sd: Dict[str, torch.Tensor],
) -> None:
    """Map a diffusers VAE AttentionProcessor to our AttentionBlock"""
    for suffix in ('weight', 'bias'):
        new_sd[f'{dst}.groupnorm.{suffix}'] = diffusers_sd[f'{src}.group_norm.{suffix}']

    for suffix in ('weight', 'bias'):
        parts = [diffusers_sd[f'{src}.{p}.{suffix}'] for p in ('to_q', 'to_k', 'to_v')]
        new_sd[f'{dst}.attention.in_proj.{suffix}'] = torch.cat(parts, dim=0)

    for suffix in ('weight', 'bias'):
        new_sd[f'{dst}.attention.out_proj.{suffix}'] = diffusers_sd[f'{src}.to_out.0.{suffix}']


def convert_vae_state_dict(diffusers_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert a diffusers AutoencoderKL state dict to our VAE format."""
    sd = {}
    for s in ('weight', 'bias'):
        sd[f'encoder.0.{s}'] = diffusers_sd[f'encoder.conv_in.{s}']

    enc_resblocks = [
        (1, 'encoder.down_blocks.0.resnets.0'),
        (2, 'encoder.down_blocks.0.resnets.1'),
        (4, 'encoder.down_blocks.1.resnets.0'),
        (5, 'encoder.down_blocks.1.resnets.1'),
        (7, 'encoder.down_blocks.2.resnets.0'),
        (8, 'encoder.down_blocks.2.resnets.1'),
        (10, 'encoder.down_blocks.3.resnets.0'),
        (11, 'encoder.down_blocks.3.resnets.1'),
        (12, 'encoder.mid_block.resnets.0'),
        (14, 'encoder.mid_block.resnets.1'),
    ]
    for idx, src in enc_resblocks:
        _map_vae_resblock(src, f'encoder.{idx}', diffusers_sd, sd)

    _map_vae_attention('encoder.mid_block.attentions.0', 'encoder.13', diffusers_sd, sd)

    for block_idx, our_idx in [(0, 3), (1, 6), (2, 9)]:
        for s in ('weight', 'bias'):
            sd[f'encoder.{our_idx}.{s}'] = diffusers_sd[f'encoder.down_blocks.{block_idx}.downsamplers.0.conv.{s}']

    for s in ('weight', 'bias'):
        sd[f'encoder.15.{s}'] = diffusers_sd[f'encoder.conv_norm_out.{s}']
        sd[f'encoder.17.{s}'] = diffusers_sd[f'encoder.conv_out.{s}']
        sd[f'encoder.18.{s}'] = diffusers_sd[f'quant_conv.{s}']

    for s in ('weight', 'bias'):
        sd[f'decoder.0.{s}'] = diffusers_sd[f'decoder.conv_in.{s}']

    _map_vae_resblock('decoder.mid_block.resnets.0', 'decoder.1', diffusers_sd, sd)
    _map_vae_attention('decoder.mid_block.attentions.0', 'decoder.2', diffusers_sd, sd)
    _map_vae_resblock('decoder.mid_block.resnets.1', 'decoder.3', diffusers_sd, sd)

    dec_resblocks = [
        (4, 'decoder.up_blocks.0.resnets.0'),
        (5, 'decoder.up_blocks.0.resnets.1'),
        (6, 'decoder.up_blocks.0.resnets.2'),
        (9, 'decoder.up_blocks.1.resnets.0'),
        (10, 'decoder.up_blocks.1.resnets.1'),
        (11, 'decoder.up_blocks.1.resnets.2'),
        (14, 'decoder.up_blocks.2.resnets.0'),
        (15, 'decoder.up_blocks.2.resnets.1'),
        (16, 'decoder.up_blocks.2.resnets.2'),
        (19, 'decoder.up_blocks.3.resnets.0'),
        (20, 'decoder.up_blocks.3.resnets.1'),
        (21, 'decoder.up_blocks.3.resnets.2'),
    ]
    for idx, src in dec_resblocks:
        _map_vae_resblock(src, f'decoder.{idx}', diffusers_sd, sd)

    for block_idx, our_idx in [(0, 8), (1, 13), (2, 18)]:
        for s in ('weight', 'bias'):
            sd[f'decoder.{our_idx}.{s}'] = diffusers_sd[f'decoder.up_blocks.{block_idx}.upsamplers.0.conv.{s}']

    for s in ('weight', 'bias'):
        sd[f'decoder.22.{s}'] = diffusers_sd[f'decoder.conv_norm_out.{s}']
        sd[f'decoder.24.{s}'] = diffusers_sd[f'decoder.conv_out.{s}']

    for s in ('weight', 'bias'):
        sd[f'post_quant_conv.{s}'] = diffusers_sd[f'post_quant_conv.{s}']

    return sd



def _map_unet_resblock(
    src: str,
    dst: str,
    diffusers_sd: Dict[str, torch.Tensor],
    new_sd: Dict[str, torch.Tensor],
) -> None:
    """Map a diffusers ResnetBlock2D to our UNet ResidualBlock."""
    mapping = {
        'norm1': 'in_layer.0',
        'conv1': 'in_layer.2',
        'time_emb_proj': 'emb_layer.1',
        'norm2': 'out_layer.0',
        'conv2': 'out_layer.3',
        'conv_shortcut': 'skip',
    }
    for src_name, dst_name in mapping.items():
        for suffix in ('weight', 'bias'):
            key = f'{src}.{src_name}.{suffix}'
            if key in diffusers_sd:
                new_sd[f'{dst}.{dst_name}.{suffix}'] = diffusers_sd[key]


def _map_spatial_transformer(
    src: str,
    dst: str,
    diffusers_sd: Dict[str, torch.Tensor],
    new_sd: Dict[str, torch.Tensor],
) -> None:
    """Map a diffusers Transformer2DModel to our SpatialTransformer."""
    for suffix in ('weight', 'bias'):
        new_sd[f'{dst}.norm.{suffix}'] = diffusers_sd[f'{src}.norm.{suffix}']
        new_sd[f'{dst}.proj_in.{suffix}'] = diffusers_sd[f'{src}.proj_in.{suffix}']
        new_sd[f'{dst}.proj_out.{suffix}'] = diffusers_sd[f'{src}.proj_out.{suffix}']

    tf_idx = 0
    while f'{src}.transformer_blocks.{tf_idx}.norm1.weight' in diffusers_sd:
        s = f'{src}.transformer_blocks.{tf_idx}'
        d = f'{dst}.blocks.{tf_idx}'

        for n in ('norm1', 'norm2', 'norm3'):
            for suffix in ('weight', 'bias'):
                new_sd[f'{d}.{n}.{suffix}'] = diffusers_sd[f'{s}.{n}.{suffix}']

        for proj in ('to_q', 'to_k', 'to_v'):
            new_sd[f'{d}.self_attn.{proj}.weight'] = diffusers_sd[f'{s}.attn1.{proj}.weight']
        new_sd[f'{d}.self_attn.to_out.weight'] = diffusers_sd[f'{s}.attn1.to_out.0.weight']
        new_sd[f'{d}.self_attn.to_out.bias'] = diffusers_sd[f'{s}.attn1.to_out.0.bias']

        for proj in ('to_q', 'to_k', 'to_v'):
            new_sd[f'{d}.cross_attn.{proj}.weight'] = diffusers_sd[f'{s}.attn2.{proj}.weight']
        new_sd[f'{d}.cross_attn.to_out.weight'] = diffusers_sd[f'{s}.attn2.to_out.0.weight']
        new_sd[f'{d}.cross_attn.to_out.bias'] = diffusers_sd[f'{s}.attn2.to_out.0.bias']

        new_sd[f'{d}.ff.geglu.proj.weight'] = diffusers_sd[f'{s}.ff.net.0.proj.weight']
        new_sd[f'{d}.ff.geglu.proj.bias'] = diffusers_sd[f'{s}.ff.net.0.proj.bias']
        new_sd[f'{d}.ff.linear.weight'] = diffusers_sd[f'{s}.ff.net.2.weight']
        new_sd[f'{d}.ff.linear.bias'] = diffusers_sd[f'{s}.ff.net.2.bias']

        tf_idx += 1


def convert_unet_state_dict(diffusers_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert a diffusers UNet2DConditionModel state dict to our UNet format."""
    sd = {}

    for s in ('weight', 'bias'):
        sd[f'down_blocks.0.0.{s}'] = diffusers_sd[f'conv_in.{s}']

    for s in ('weight', 'bias'):
        sd[f'ts_embedding.0.{s}'] = diffusers_sd[f'time_embedding.linear_1.{s}']
        sd[f'ts_embedding.2.{s}'] = diffusers_sd[f'time_embedding.linear_2.{s}']

    down_map = [
        (1, 0, 0, True),
        (2, 0, 1, True),
        (3, 0, -1, False),
        (4, 1, 0, True),
        (5, 1, 1, True),
        (6, 1, -1, False),
        (7, 2, 0, True),
        (8, 2, 1, True),
        (9, 2, -1, False),
        (10, 3, 0, False),
        (11, 3, 1, False),
    ]

    for our_idx, block, res_idx, has_attn in down_map:
        if res_idx == -1:
            for s in ('weight', 'bias'):
                sd[f'down_blocks.{our_idx}.0.conv.{s}'] = \
                    diffusers_sd[f'down_blocks.{block}.downsamplers.0.conv.{s}']
        else:
            _map_unet_resblock(
                f'down_blocks.{block}.resnets.{res_idx}',
                f'down_blocks.{our_idx}.0',
                diffusers_sd, sd,
            )
            if has_attn:
                _map_spatial_transformer(
                    f'down_blocks.{block}.attentions.{res_idx}',
                    f'down_blocks.{our_idx}.1',
                    diffusers_sd, sd,
                )

    _map_unet_resblock('mid_block.resnets.0', 'middle_blocks.0', diffusers_sd, sd)
    _map_spatial_transformer('mid_block.attentions.0', 'middle_blocks.1', diffusers_sd, sd)
    _map_unet_resblock('mid_block.resnets.1', 'middle_blocks.2', diffusers_sd, sd)

    up_map = [
        (0, 0, 0, False, False),
        (1, 0, 1, False, False),
        (2, 0, 2, False, True),
        (3, 1, 0, True, False),
        (4, 1, 1, True, False),
        (5, 1, 2, True, True),
        (6, 2, 0, True, False),
        (7, 2, 1, True, False),
        (8, 2, 2, True, True),
        (9, 3, 0, True, False),
        (10, 3, 1, True, False),
        (11, 3, 2, True, False),
    ]

    for our_idx, block, res_idx, has_attn, has_up in up_map:
        _map_unet_resblock(
            f'up_blocks.{block}.resnets.{res_idx}',
            f'up_blocks.{our_idx}.0',
            diffusers_sd, sd,
        )
        sub = 1
        if has_attn:
            _map_spatial_transformer(
                f'up_blocks.{block}.attentions.{res_idx}',
                f'up_blocks.{our_idx}.{sub}',
                diffusers_sd, sd,
            )
            sub += 1
        if has_up:
            for s in ('weight', 'bias'):
                sd[f'up_blocks.{our_idx}.{sub}.conv.{s}'] = \
                    diffusers_sd[f'up_blocks.{block}.upsamplers.0.conv.{s}']

    for s in ('weight', 'bias'):
        sd[f'out.0.{s}'] = diffusers_sd[f'conv_norm_out.{s}']
        sd[f'out.2.{s}'] = diffusers_sd[f'conv_out.{s}']

    return sd
