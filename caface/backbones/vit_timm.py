from enum import Enum
import sys


import timm
import torch
import torch.nn as nn

if __name__ == "__main__":
    sys.path.append(
    "/lustre/scratch/client/vinai/users/trungdt21/workspace/matching/code/insightface/recognition/arcface_torch"
    )
from backbones.layers import AdaptiveConcatPool1d


class ViTHeadType(Enum):
    DENSE = "dense"
    DENSE_NORM = "dense_norm"
    BLOCK_OF_DENSE = "block_of_dense"


class ViTNeckType(Enum):
    IDENTITY = "identity"
    MAXPOOL = "maxpool"
    AVGPOOL = "avgpool"
    CONCATPOOL = "concatpool"


class ViTFeatureType(Enum):
    CLS = "cls"
    N_TOKENS_AVG = "n_tokens_avg"
    N_TOKENS_MAX = "n_tokens_max"
    N_TOKENS_FLATTEN = "n_tokens_flatten"


class VitTIMM(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        timm_dict = {
            "model_name": config.model_name,
            "pretrained": config.pretrained,
            "img_size": config.img_size,
            "num_classes": 0,
            "global_pool": "",
        }
        if config.patch_size is not None:
            timm_dict["patch_size"] = config.patch_size
        if config.encoder_embed_dim is not None:
            timm_dict["encoder_embed_dim"] = config.encoder_embed_dim
        if config.model_name != "timm_custom":
            self.encoder = timm.create_model(**timm_dict)
        else:
            # TODO: create custom ViT model
            pass


        self.encoder_embed_dim = getattr(self.encoder, "embed_dim", config.neck_size)
        self.final_embed_dim = config.final_embed_dim

        if config.feature_type != "cls" and config.neck_type == "identity":
            raise Exception("Invalid combination configuration: cls + identity")

        self.neck_size = (
            self.encoder_embed_dim
            if config.neck_type == ViTNeckType.IDENTITY.value
            else config.neck_size
        )
        self.feature_type = config.feature_type

        self.neck = self.create_neck(config.neck_type, self.neck_size)
        self.head = self.create_head(config.head_type, self.neck_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.get_encoder_feature(x, self.feature_type)
        x = self.neck(x)
        x = self.head(x)
        return x

    def create_neck(self, type: ViTNeckType, neck_size: int = None):
        if type == ViTNeckType.IDENTITY.value:
            return nn.Identity()
        elif type == ViTNeckType.MAXPOOL.value:
            return nn.AdaptiveMaxPool1d(neck_size)
        elif type == ViTNeckType.AVGPOOL.value:
            return nn.AdaptiveAvgPool1d(neck_size)
        elif type == ViTNeckType.CONCATPOOL.value:
            return AdaptiveConcatPool1d(neck_size)
        else:
            raise Exception(f"Unknown neck type %s" % type)

    def create_head(self, type: ViTHeadType, input_sz: int = None):
        if type == ViTHeadType.DENSE.value:
            return nn.Linear(input_sz, self.final_embed_dim)
        elif type == ViTHeadType.DENSE_NORM.value:
            return nn.Sequential(
                nn.LayerNorm(input_sz, eps=1e-6),
                nn.Linear(input_sz, self.final_embed_dim),
            )
        elif type == ViTHeadType.BLOCK_OF_DENSE.value:
            return nn.Sequential(
                nn.Linear(
                    in_features=input_sz,
                    out_features=self.final_embed_dim * 2,
                    bias=False,
                ),
                nn.BatchNorm1d(num_features=self.final_embed_dim * 2, eps=2e-5),
                nn.Linear(
                    in_features=self.final_embed_dim * 2, out_features=512, bias=False
                ),
                nn.BatchNorm1d(num_features=512, eps=2e-5),
            )
        else:
            raise Exception(f"Unknown head type %s" % type)

    def get_encoder_feature(self, x: torch.Tensor, type: ViTFeatureType):
        if type == ViTFeatureType.CLS.value:
            return x[:, 0]
        elif type == ViTFeatureType.N_TOKENS_AVG.value:
            return x[:, 1:].mean(dim=1)
        elif type == ViTFeatureType.N_TOKENS_MAX.value:
            return x[:, 1:].max(dim=1)
        elif type == ViTFeatureType.N_TOKENS_FLATTEN.value:
            return x[:, 1:].reshape(x.shape[0], -1)
        else:
            raise Exception(f"Unknown feature type %s" % type)


if __name__ == "__main__":
    from configs.custom_configs.wf42m_raw_pfc03_40epoch_16gpu_mobilevit_timm import config
    from rich import print

    print(config.timm)
    m = VitTIMM(config.timm)
    input = torch.rand((4, 3, 112, 112))
    f = m(input)
    print(f.shape)
