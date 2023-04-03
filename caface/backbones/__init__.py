
def get_model(name, **kwargs):
    # resnet
    export_onnx = kwargs.get("export_onnx", False)

    if name == "vit_t":
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
