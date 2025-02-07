import numpy as np
import torch
from tqdm import tqdm


def infer_features(dataloader, model, aggregator, hyper_param, device="cuda:0"):
    features = []
    norms = []
    intermediates = []
    prev_max_idx = 0
    with torch.no_grad():
        for iter_idx, (img, idx) in tqdm(enumerate(dataloader), total=len(dataloader)):
            assert idx.max().item() >= prev_max_idx
            prev_max_idx = idx.max().item()  # order shifting by dataloader checking

            if hyper_param.intermediate_type == "map":
                feature, norm, intermediate = model(
                    img.to(device), return_intermediate=True
                )
                compressed_intermediate = aggregator.compress_intermediate(intermediate)
            elif hyper_param.intermediate_type == "style":
                feature, norm, intermediate = model(
                    img.to(device), return_style=hyper_param.style_index
                )
                compressed_intermediate = aggregator.compress_intermediate(intermediate)
            else:
                raise ValueError("not a correct intermediate")

            feature = feature * norm
            features.append(feature.cpu().numpy())
            norms.append(norm.cpu().numpy())
            intermediates.append(compressed_intermediate.cpu().numpy())

    features = np.concatenate(features, axis=0)
    intermediates = np.concatenate(intermediates, axis=0)
    return features, intermediates


def l2_normalize(x, axis=1, eps=1e-8):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def fuse_feature(
    features,
    aggregator=None,
    intermediates=None,
    method="cluster_and_aggregate",
    running_avg_alpha=0.5,
    scores=None,
    device="cuda:0",
):
    if len(features) == 1:
        fused = features[0]
        weights = np.ones(1)
        return fused, weights

    if method == "cluster_and_aggregate":
        fused, weights = aggregator_caface_fuse(
            aggregator=aggregator,
            features=torch.tensor(features).float().to(device).unsqueeze(0),
            intermediate=torch.tensor(intermediates).float().to(device).unsqueeze(0),
            max_element=512,
        )
    elif method == "norm_weight":
        fused = np.zeros(features.shape[1])
        weights = (
            np.linalg.norm(features, 2, -1) / np.linalg.norm(features, 2, -1).sum()
        )
        for feat, w in zip(features, weights):
            fused += w * feat
        fused = l2_normalize(fused, -1)
    elif method == "average":
        fused = l2_normalize(features.mean(0), -1)
        weights = np.ones(len(features)) / len(features)
    elif method == "running_avg":
        fused = features[0]
        for idx in range(1, features.shape[0]):
            fused = (1 - running_avg_alpha) * fused + running_avg_alpha * features[idx]
        fused = l2_normalize(fused, -1)
        weights = np.ones(len(features)) / len(features)
    elif method == "naive":
        fused = l2_normalize(features[-1], -1)
        weights = np.ones(len(features)) / len(features)
    elif method == "scores_txt":
        assert scores is not None, "Scores array should not be None"
        fused = np.zeros(features.shape[1])
        weights = scores
        for feat, w in zip(features, weights):
            fused += w * feat
        fused = l2_normalize(fused, -1)
    else:
        raise ValueError("not a correct value for fusion method")

    return fused, weights


def aggregator_caface_fuse(aggregator, features, intermediate, max_element=512):
    # perform Cluster and Aggregate feature fusion
    # max_element: divide N features into batches of size 128
    # and sequentially update

    aggregator.use_memory = True

    def l2_normalize(x, axis=1, eps=1e-8):
        return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)

    def infer_with(aggregator, features, interms):
        agg_result = aggregator(features, interms)

        weights = agg_result["confidence_weight"].detach().cpu().numpy()
        assignment = (
            agg_result["attn_dict_list"][0]["attn"][0][0].detach().cpu().numpy()
        )
        # influence = assignment / assignment.sum(-1, keepdims=True)
        # weights = (influence * np.expand_dims(weights.mean(-1)[0], -1)).sum(0)

        aggregated = agg_result["aggregated"]
        aggregated_np = aggregated.squeeze(0).detach().cpu().numpy()
        return aggregated_np, weights, assignment

    if features.shape[1] > max_element:
        T = features.shape[1]
        num_split = int(np.ceil(T / max_element))
        splits = np.array_split(np.arange(T), num_split)
        fused = None
        weights = []
        assignment = []
        with torch.no_grad():
            for split_idx, split in enumerate(splits):
                # try:
                subfeature = features[:, split, :]
                subinterm = intermediate[:, split, :]
                nan_index = subfeature.isnan().any(0).any(-1)
                subfeature = subfeature[:, ~nan_index, :]
                subinterm = subinterm[:, ~nan_index, :]
                if split_idx == 0:
                    aggregator.memory_count = 0
                fused, _weights, _assignment = infer_with(
                    aggregator, subfeature, subinterm
                )
                weights.append(_weights)
                assignment.append(_assignment)
        fused_np = l2_normalize(fused, axis=0)
        weights = weights[-1]
        assignment = np.concatenate(assignment, axis=1)
        influence = assignment / assignment.sum(-1, keepdims=True)
        weights = (influence * np.expand_dims(weights.mean(-1)[0], -1)).sum(0)
        aggregator.memory_count = 0
    else:
        aggregator.memory_count = 0
        fused, weights, assignment = infer_with(aggregator, features, intermediate)
        influence = assignment / assignment.sum(-1, keepdims=True)
        weights = (influence * np.expand_dims(weights.mean(-1)[0], -1)).sum(0)
        fused_np = l2_normalize(fused, axis=0)
        aggregator.memory_count = 0

    return fused_np, weights
