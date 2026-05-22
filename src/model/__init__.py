import attr

from .vgg import VGG
from .vit_small import ViT_Small


SUPPORTED_MODELS = ("vgg19", "vit_small")


@attr.s
class ModelSetting:
    name = attr.ib()
    num_classes = attr.ib()


def build_model(setting: ModelSetting):
    if setting.name == "vgg19":
        return VGG("VGG19", num_classes=setting.num_classes)

    if setting.name == "vit_small":
        return ViT_Small(
            image_size=32,
            patch_size=4,
            num_classes=setting.num_classes,
            dim=512,
            depth=6,
            heads=8,
            mlp_dim=512,
            dropout=0,
            emb_dropout=0,
        )

    raise ValueError(
        f"Unsupported model '{setting.name}'. Supported models: {', '.join(SUPPORTED_MODELS)}"
    )
