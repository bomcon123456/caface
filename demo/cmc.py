import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from natsort import natsorted
from prettytable import PrettyTable

sys.path.append("/home/termanteus/workspace/face/code/caface/demo")
sys.path.append("/home/termanteus/workspace/face/code/caface")

from plotcmc import CMC

CACHE_FOLDER = Path("./cache")
CACHE_FOLDER.mkdir(parents=True, exist_ok=True)

import visualization
from dataset import prepare_imagelist_dataloader, to_tensor
from face_alignment.aligner import FaceAligner
from face_detection.detector import FaceDetector
from inference import fuse_feature, infer_features, l2_normalize
from model_loader import load_caface

app = typer.Typer()


@app.command()
def eval(
    gallery_path: Path = typer.Argument(..., help="Path to gallery images"),
    mate_path: Path = typer.Argument(..., help="Path to mate images"),
    ckpt_path: Path = typer.Argument(..., help="Path to checkpoint"),
    output_dir: Path = typer.Argument(..., help="Path to ouput"),
    feature_size: int = typer.Option(512, help="Feature size"),
    device: str = typer.Option("cuda:0", help="Device"),
    # 'cluster_and_aggregate', 'norm_weight', 'average'
    fusion_method: str = typer.Option("average", help="fusion method"),
    running_avg_alpha: float = typer.Option(0.5, help="running avg alpha"),
):
    basename = f"{ckpt_path.stem}/{gallery_path.stem}_CMC_{mate_path.stem}"
    output_dir = output_dir / f"{basename}"
    output_dir.mkdir(parents=True, exist_ok=True)
    gallery_ids = os.listdir(gallery_path)
    gallery_ids = natsorted(
        filter(lambda x: len(list(os.listdir(gallery_path / x))) > 0, gallery_ids)
    )
    gallery_ids = list(map(lambda x: x.strip().lower(), gallery_ids))
    gallery_features = np.zeros((len(gallery_ids), feature_size))
    print("Gallery size: ", len(gallery_ids))

    detector = FaceDetector()
    aligner = FaceAligner()
    # load caface
    aggregator, model, hyper_param = load_caface(ckpt_path, device=device)

    gallery_cache = CACHE_FOLDER / f"{basename}/gallery.npy"
    matefeats_nonfused_cache = CACHE_FOLDER / f"{basename}/mate_features.npy"
    mateints_nonfused_cache = CACHE_FOLDER / f"{basename}/mate_intermediates.npy"
    gallery_cache.parent.mkdir(exist_ok=True, parents=True)

    if gallery_cache.exists():
        gallery_features = np.load(gallery_cache)
    else:
        for gallery_idx, gallery_id in enumerate(gallery_ids):
            id_path = gallery_path / gallery_id

            gallery_images = list(id_path.glob("*.[jp][pn]g"))
            if len(gallery_images) == 0:
                print(f"{gallery_id} has 0 images")
                continue
            fs = []
            for gallery_image_path in gallery_images:
                gallery_image = cv2.imread(gallery_image_path.as_posix())
                if gallery_image.shape[0] == 112 and gallery_image.shape[1] == 112:
                    pass
                else:
                    gallery_image = aligner.align(detector.detect(gallery_image))
                gallery_image_tensor = to_tensor(gallery_image, device=device)
                with torch.no_grad():
                    gallery_feature, _ = model(gallery_image_tensor)
                gallery_feature = gallery_feature.detach().cpu().numpy()
                fs.append(gallery_feature)
            fs = np.array(fs)
            gallery_feature = l2_normalize(np.array(fs).mean(0))
            gallery_features[gallery_idx] = gallery_feature
        np.save(gallery_cache, gallery_features)

    print("Gallery feature shape: ", gallery_features.shape)
    mate_folders = os.listdir(mate_path)
    final_mate_folders = []
    mate_not_in_gallery = set()
    for mate_folder in mate_folders:
        curmate_path = mate_path / mate_folder
        curmate_images = list(curmate_path.glob("*.[jp][pn]g"))
        if len(curmate_images) == 0:
            continue
        mate_id = mate_folder.split("_")[-1].strip().lower()
        if mate_id not in gallery_ids:
            mate_not_in_gallery.add(mate_id)
            continue
        final_mate_folders.append(mate_folder)
    print("gallery ids: ", gallery_ids)
    print("List mate that doesn't have gallery, will be skipped: ", mate_not_in_gallery)
    final_mate_folders = natsorted(final_mate_folders)
    if matefeats_nonfused_cache.exists():
        mate_nonfused_features = np.load(matefeats_nonfused_cache, allow_pickle=True)
        mate_nonfused_intermediates = np.load(
            mateints_nonfused_cache, allow_pickle=True
        )
        assert len(mate_nonfused_features) == len(mate_nonfused_intermediates)

        mate_features = np.zeros((len(final_mate_folders), feature_size))
        for mate_idx, (probe_features, probe_intermediates) in enumerate(
            zip(mate_nonfused_features, mate_nonfused_intermediates)
        ):
            probe_fused_feature, _ = fuse_feature(
                probe_features,
                aggregator,
                probe_intermediates,
                method=fusion_method,
                device=device,
                running_avg_alpha=running_avg_alpha,
            )
            mate_features[mate_idx] = probe_fused_feature

    else:
        mate_nonfused_features = []
        mate_nonfused_intermediates = []
        mate_features = np.zeros((len(final_mate_folders), feature_size))
        for mate_idx, mate_id in enumerate(final_mate_folders):
            id_path = mate_path / mate_id

            mate_images = list(id_path.glob("*.[jp][pn]g"))
            dataloader = prepare_imagelist_dataloader(
                mate_images, batch_size=16, num_workers=0
            )
            probe_features, probe_intermediates = infer_features(
                dataloader, model, aggregator, hyper_param, device=device
            )
            mate_nonfused_features.append(probe_features)
            mate_nonfused_intermediates.append(probe_intermediates)
            probe_fused_feature, _ = fuse_feature(
                probe_features,
                aggregator,
                probe_intermediates,
                method=fusion_method,
                device=device,
                running_avg_alpha=running_avg_alpha,
            )
            mate_features[mate_idx] = probe_fused_feature
        mate_nonfused_features = np.array(mate_nonfused_features)
        mate_nonfused_intermediates = np.array(mate_nonfused_intermediates)
        np.save(matefeats_nonfused_cache, mate_nonfused_features)
        np.save(mateints_nonfused_cache, mate_nonfused_intermediates)

    mate_gallery_similarity = np.dot(mate_features, gallery_features.T)
    mate_labels = np.array(
        [mate_id.split("_")[-1].strip().lower() for mate_id in final_mate_folders]
    )
    num_mate_searchs = len(final_mate_folders)

    all_mate_predict_indices = np.argsort(-mate_gallery_similarity, axis=1)
    all_mate_predict_idstr = []
    for mate_predict_indices in all_mate_predict_indices:
        strs = [gallery_ids[idx] for idx in mate_predict_indices]
        all_mate_predict_idstr.append(strs)
    res = {}
    for rank in range(1, 21):
        TP = 0
        for mate_idx, mate_preds_str_ in enumerate(all_mate_predict_idstr):
            mate_preds_str = mate_preds_str_[:rank]
            mate_lbl = mate_labels[mate_idx]
            if mate_lbl in mate_preds_str:
                TP += 1
        res[rank] = TP / num_mate_searchs

    cmc = CMC({fusion_method: list(res.values())})
    cmc.save(title="CMC", filename=f"{output_dir}/cmc_plot_{fusion_method}")
    with open(f"{output_dir}/cmc_{fusion_method}.json", "w") as f:
        json.dump(res, f, indent=2)

    return res


@app.command()
def plot(
    output_dir: Path = typer.Argument(..., help="Output dir"),
    exps: List[Path] = typer.Option([], help="Experiment folders"),
    override_legend: Optional[List[str]] = typer.Option(
        None, "--override", "-o", help="default will be exp folder name"
    ),
):
    if override_legend is None or len(override_legend) == 0:
        override_legend = [exp.stem for exp in exps]
    d = {}
    for exp, legend_name in zip(exps, override_legend):
        with open(exp, "r") as f:
            vals = list(json.load(f).values())
        d[legend_name] = vals
    cmc = CMC(d)
    cmc.save(title="CMC", filename=f"{output_dir}/cmc_all")


if __name__ == "__main__":
    app()
