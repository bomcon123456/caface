from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200
from .resnet import ResNet_50, ResNet_101, ResNet_152, ResNet_200
from .mobilefacenet import get_mbf


def get_model(name, **kwargs):
    # resnet
    export_onnx = kwargs.get("export_onnx", False)
    if name == "r18":
        return iresnet18(False)
    elif name == "r34":
        return iresnet34(False)
    elif name == "r50":
        return iresnet50(False)
    elif name == "r100":
        return iresnet100(False)
    elif name == "r200":
        return iresnet200(False)
    elif name == "r2060":
        from .iresnet2060 import iresnet2060
        return iresnet2060(False)
    elif name == "resnet50":
        return ResNet_50([112,112], export_onnx=export_onnx)
    elif name == "resnet101":
        return ResNet_101([112,112], export_onnx=export_onnx)
    elif name == "resnet152":
        return ResNet_152([112,112], export_onnx=export_onnx)
    elif name == "resnet200":
        return ResNet_200([112,112], export_onnx=export_onnx)

    elif name == "mbf":
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        return get_mbf(fp16=fp16, num_features=num_features)

    elif name == "mbf_large":
        from .mobilefacenet import get_mbf_large
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        return get_mbf_large(fp16=fp16, num_features=num_features)

    elif name == "vit_t":
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=256, depth=12,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1)

    elif name == "vit_t_dp005_mask0": # For WebFace42M
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=256, depth=12,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.0)

    elif name == "vit_s":
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=12,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1)
    
    elif name == "vit_s_dp005_mask_0":  # For WebFace42M
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=12,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.0)
    
    elif name == "vit_b":
        # this is a feature
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=24,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1, using_checkpoint=True)

    elif name == "vit_b_dp005_mask_005":  # For WebFace42M
        # this is a feature
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=24,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.05, using_checkpoint=True)

    elif name == "vit_l_dp005_mask_005":  # For WebFace42M
        # this is a feature
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=768, depth=24,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.05, using_checkpoint=True)
    elif name == "vit_l_dp005_mask_005_qc":  # For WebFace42M
        # this is a feature
        num_features = kwargs.get("num_features", 512)
        from .vit_qc import VisionTransformerQC
        return VisionTransformerQC(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=768, depth=24,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.05, using_checkpoint=True)
    elif name == "vit_l_p16_dp005_mask_005":  # For WebFace42M
        # this is a feature
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=16, num_classes=num_features, embed_dim=768, depth=24,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.05, using_checkpoint=True)
    elif name == "vit_l_p16_dp005_mask_005_qc":  # For WebFace42M
        # this is a feature
        num_features = kwargs.get("num_features", 512)
        from .vit_qc import VisionTransformerQC
        return VisionTransformerQC(
            img_size=112, patch_size=16, num_classes=num_features, embed_dim=768, depth=24,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.05, using_checkpoint=True)
    elif name == "timm":  # For WebFace42M
        cfg = kwargs.get("cfg")
        from .vit_timm import VitTIMM
        return VitTIMM(cfg.timm)
    elif name == "timmsimp":  # For WebFace42M
        cfg = kwargs.get("cfg")
        from .timm_simple import TIMMSimple
        return TIMMSimple(cfg.timm)
    elif name == "timm_vit_large_patch16_224":  # For WebFace42M
        num_features = kwargs.get("num_features", 512)
        import timm
        return timm.create_model(name.replace("timm_",""), pretrained=False, img_size=112, num_classes=num_features)
    elif name == "timm_vit_base_patch16_384":  # For WebFace42M
        num_features = kwargs.get("num_features", 512)
        import timm
        return timm.create_model(name.replace("timm_",""), pretrained=False, img_size=112, num_classes=num_features)
    elif "timm_mobilevit" in name:  # For WebFace42M
        num_features = kwargs.get("num_features", 512)
        import timm
        return timm.create_model(name.replace("timm_",""), pretrained=False, img_size=112, num_classes=num_features)
    else:
        raise ValueError()
