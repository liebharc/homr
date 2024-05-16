from typing import Any

from timm.models.layers import StdConv2dSame  # type: ignore
from timm.models.resnetv2 import ResNetV2  # type: ignore
from timm.models.vision_transformer import VisionTransformer  # type: ignore
from timm.models.vision_transformer_hybrid import HybridEmbed  # type: ignore

from homr.transformer.configs import Config


def get_encoder(config: Config) -> Any:
    backbone_layers = list(config.backbone_layers)
    backbone = ResNetV2(
        num_classes=0,
        global_pool="",
        in_chans=config.channels,
        drop_rate=0.1,
        drop_path_rate=0.1,
        layers=backbone_layers,
        preact=True,
        stem_type="same",
        conv_layer=StdConv2dSame,
    )
    min_patch_size = 2 ** (len(backbone_layers) + 1)

    def embed_layer(**x: Any) -> Any:
        ps = x.pop("patch_size", min_patch_size)
        if ps % min_patch_size != 0 or ps < min_patch_size:
            raise ValueError(
                "patch_size needs to be multiple of %i with current backbone configuration"
                % min_patch_size
            )
        return HybridEmbed(**x, patch_size=ps // min_patch_size, backbone=backbone)

    encoder = VisionTransformer(
        img_size=(config.max_height, config.max_width),
        patch_size=config.patch_size,
        in_chans=config.channels,
        num_classes=0,
        embed_dim=config.encoder_dim,
        depth=config.encoder_depth,
        num_heads=config.encoder_heads,
        embed_layer=embed_layer,
        global_pool="",
    )
    return encoder
