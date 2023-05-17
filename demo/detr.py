import os
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from tqdm import tqdm
from natsort import natsorted
from prettytable import PrettyTable

sys.path.append("/home/termanteus/workspace/face/code/caface/demo")
sys.path.append("/home/termanteus/workspace/face/code/caface")

CACHE_FOLDER = Path("./cache")
CACHE_FOLDER.mkdir(parents=True, exist_ok=True)

# fmt: off
import visualization
from dataset import prepare_imagelist_dataloader, to_tensor
from face_alignment.aligner import FaceAligner
from face_detection.detector import FaceDetector
from inference import fuse_feature, infer_features, l2_normalize
from model_loader import load_caface

# fmt: on

app = typer.Typer()


@app.command()
def eval(
    gallery_path: Path = typer.Argument(..., help="Path to gallery images"),
    mate_path: Path = typer.Argument(..., help="Path to mate images"),
    nonmate_path: Path = typer.Argument(..., help="Path to nonmate images"),
    ckpt_path: Path = typer.Argument(..., help="Path to checkpoint"),
    output_dir: Path = typer.Argument(..., help="Path to ouput"),
    feature_size: int = typer.Option(512, help="Feature size"),
    steps: int = typer.Option(1000, help="threshold step"),
    device: str = typer.Option("cuda:0", help="Device"),
    # 'cluster_and_aggregate', 'norm_weight', 'average',
    fusion_method: str = typer.Option("average", help="fusion method"),
    running_avg_alpha: float = typer.Option(0.5, help="running avg alpha"),
):
    output_dir = (
        output_dir
        / f"{ckpt_path.stem}/{gallery_path.stem}_{mate_path.stem}_{nonmate_path.stem}/{fusion_method}"
    )
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

    gallery_cache = (
        CACHE_FOLDER
        / f"{ckpt_path.stem}/{gallery_path.stem}_{mate_path.stem}_{nonmate_path.stem}/gallery.npy"
    )
    matefeats_nonfused_cache = (
        CACHE_FOLDER
        / f"{ckpt_path.stem}/{gallery_path.stem}_{mate_path.stem}_{nonmate_path.stem}/mate_features.npy"
    )
    mateints_nonfused_cache = (
        CACHE_FOLDER
        / f"{ckpt_path.stem}/{gallery_path.stem}_{mate_path.stem}_{nonmate_path.stem}/mate_intermediates.npy"
    )
    nonmatefeats_nonfused_cache = (
        CACHE_FOLDER
        / f"{ckpt_path.stem}/{gallery_path.stem}_{mate_path.stem}_{nonmate_path.stem}/nonmate_features.npy"
    )
    nonmateints_nonfused_cache = (
        CACHE_FOLDER
        / f"{ckpt_path.stem}/{gallery_path.stem}_{mate_path.stem}_{nonmate_path.stem}/nonmate_intermediates.npy"
    )
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

    nonmate_ids = os.listdir(nonmate_path)
    nonmate_ids = natsorted(
        filter(lambda x: len(list(os.listdir(nonmate_path / x))) > 0, nonmate_ids)
    )
    if nonmatefeats_nonfused_cache.exists():
        nonmate_nonfused_features = np.load(
            nonmatefeats_nonfused_cache, allow_pickle=True
        )
        nonmate_nonfused_intermediates = np.load(
            nonmateints_nonfused_cache, allow_pickle=True
        )
        assert len(nonmate_nonfused_features) == len(nonmate_nonfused_intermediates)

        nonmate_features = np.zeros((len(nonmate_ids), feature_size))
        for nonmate_idx, (probe_features, probe_intermediates) in enumerate(
            zip(nonmate_nonfused_features, nonmate_nonfused_intermediates)
        ):
            probe_fused_feature, _ = fuse_feature(
                probe_features,
                aggregator,
                probe_intermediates,
                method=fusion_method,
                device=device,
                running_avg_alpha=running_avg_alpha,
            )
            nonmate_features[nonmate_idx] = probe_fused_feature
    else:
        nonmate_features = np.zeros((len(nonmate_ids), feature_size))
        nonmate_nonfused_features = []
        nonmate_nonfused_intermediates = []
        for nonmate_idx, nonmate_id in enumerate(nonmate_ids):
            id_path = nonmate_path / nonmate_id

            nonmate_images = list(id_path.glob("*.[jp][pn]g"))
            dataloader = prepare_imagelist_dataloader(
                nonmate_images, batch_size=16, num_workers=0
            )
            probe_features, probe_intermediates = infer_features(
                dataloader, model, aggregator, hyper_param, device=device
            )
            nonmate_nonfused_features.append(probe_features)
            nonmate_nonfused_intermediates.append(probe_intermediates)
            probe_fused_feature, _ = fuse_feature(
                probe_features,
                aggregator,
                probe_intermediates,
                method=fusion_method,
                device=device,
                running_avg_alpha=running_avg_alpha,
            )
            nonmate_features[nonmate_idx] = probe_fused_feature
        np.save(nonmatefeats_nonfused_cache, nonmate_nonfused_features)
        np.save(nonmateints_nonfused_cache, nonmate_nonfused_intermediates)
        # np.save(nonmate_cache, nonmate_features)

    mate_gallery_similarity = np.dot(mate_features, gallery_features.T)
    nonmate_gallery_similarity = np.dot(nonmate_features, gallery_features.T)

    mate_predict_idx, mate_predict_score = np.argmax(
        mate_gallery_similarity, axis=1
    ), np.max(mate_gallery_similarity, axis=1)
    _, nonmate_predict_score = np.argmax(nonmate_gallery_similarity, axis=1), np.max(
        nonmate_gallery_similarity, axis=1
    )
    mate_predict_str = np.array([gallery_ids[idx] for idx in mate_predict_idx])
    mate_label = np.array(
        [mate_id.split("_")[-1].strip().lower() for mate_id in final_mate_folders]
    )

    num_mate_searchs = len(final_mate_folders)
    num_nonmate_searchs = len(nonmate_ids)

    fpirs, fnirs = [], []
    for threshold in np.arange(0, 1, 1 / steps):
        valid_ids = mate_predict_score >= threshold
        valid_predict = mate_predict_str[valid_ids]
        valid_label = mate_label[valid_ids]
        TP = np.sum(valid_predict == valid_label)
        FN = num_mate_searchs - TP

        nonmate_accept_ids = nonmate_predict_score >= threshold
        FP = np.sum(nonmate_accept_ids)

        # Calculate FPIR
        FPIR = FP / num_mate_searchs
        # Calculate FNIR
        FNIR = FN / num_nonmate_searchs
        fpirs.append(FPIR)
        fnirs.append(FNIR)
        if threshold in [0.3, 0.4, 0.5]:
            print(f"TP@{threshold}: {TP}")
            print(f"FP@{threshold}: {FP}")
            print(f"FN@{threshold}: {FN}")
            print(f"FPIR@{threshold}: {FPIR}")
            print(f"FNIR@{threshold}: {FNIR}")

    plt.figure(figsize=(15, 7))
    # plt.scatter(fnirs, fpirs, s=100, alpha=0.5, color="blue", label="Scikit-learn")
    plt.plot(fpirs, fnirs, linestyle="solid")
    plt.title(
        f"DET Curve \n #mate-search: {num_mate_searchs}; #nonmate-search: {num_nonmate_searchs}",
        fontsize=20,
    )
    plt.xlabel("False Positive Identification Rate", fontsize=16)
    plt.ylabel("False Negative Identification Rate", fontsize=16)
    plt.savefig((output_dir / f"det_curve.png").as_posix())

    x_labels = [10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1]
    fnir_fpir_table = PrettyTable(["Methods"] + [str(x) for x in x_labels])
    fnir_fpir_row = []
    fnir_fpir_row.append("Model/Metrics")
    fpirs = np.array(fpirs)
    for fpir_iter in np.arange(len(x_labels)):
        _, min_index = min(
            list(zip(abs(fpirs - x_labels[fpir_iter]), range(len(fpirs))))
        )
        # tpr_fpr_row.append('%.4f' % tpr[min_index])
        fnir_fpir_row.append("%.6f" % fnirs[min_index])
    fnir_fpir_table.add_row(fnir_fpir_row)
    print(fnir_fpir_table)
    with open((output_dir / f"table.txt").as_posix(), "w") as f:
        print(fnir_fpir_table, file=f)
    np.save((output_dir / f"fpir.npy").as_posix(), fpirs)
    np.save((output_dir / f"fnir.npy").as_posix(), fnirs)

@app.command()
def eval_multiple_gallery(
    mate_path: Path = typer.Argument(..., help="Path to mate images"),
    nonmate_path: Path = typer.Argument(..., help="Path to nonmate images"),
    ckpt_path: Path = typer.Argument(..., help="Path to checkpoint"),
    output_dir: Path = typer.Argument(..., help="Path to ouput"),
    feature_size: int = typer.Option(512, help="Feature size"),
    steps: int = typer.Option(1000, help="threshold step"),
    device: str = typer.Option("cuda:0", help="Device"),
    # 'cluster_and_aggregate', 'norm_weight', 'average',
    fusion_method: str = typer.Option("average", help="fusion method"),
    running_avg_alpha: float = typer.Option(0.5, help="running avg alpha"),
    gallery_paths: List[Path] = typer.Option([], help="Path to gallery images"),
    gallery_size: str = typer.Option("150,250,500,1000,2500,5000,10000,15000,20000,25000", help="gallery step")
):
    assert len(gallery_paths) > 0, "Gallery must not be empty!"
    gallery_joint_name = "_".join(list(map(lambda x: x.stem, gallery_paths)))
    exp_dir = f"{ckpt_path.stem}/{gallery_joint_name}_{mate_path.stem}_{nonmate_path.stem}"
    output_dir = (
        output_dir
        / f"{exp_dir}/{fusion_method}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    gallery_ids = []
    gallery_id_paths = []
    for gallery_path in gallery_paths:
        gallery_ids_tmp = os.listdir(gallery_path)
        gallery_ids_tmp = natsorted(
            filter(lambda x: len(list(os.listdir(gallery_path / x))) > 0, gallery_ids_tmp)
        )
        gallery_ids_tmp = list(map(lambda x: x.strip().lower(), gallery_ids_tmp))
        gallery_id_path_tmp = list(map(lambda x: gallery_path / x, gallery_ids_tmp))
        gallery_id_paths.extend(gallery_id_path_tmp)
        gallery_ids.extend(gallery_ids_tmp)

    gallery_features = np.zeros((len(gallery_ids), feature_size))
    print("Gallery original size: ", len(gallery_ids))

    detector = FaceDetector()
    aligner = FaceAligner()
    # load caface
    aggregator, model, hyper_param = load_caface(ckpt_path, device=device)

    gallery_cache = (
        CACHE_FOLDER
        / f"{exp_dir}/gallery.npy"
    )
    matefeats_nonfused_cache = (
        CACHE_FOLDER
        / f"{exp_dir}/mate_features.npy"
    )
    mateints_nonfused_cache = (
        CACHE_FOLDER
        / f"{exp_dir}/mate_intermediates.npy"
    )
    nonmatefeats_nonfused_cache = (
        CACHE_FOLDER
        / f"{exp_dir}/nonmate_features.npy"
    )
    nonmateints_nonfused_cache = (
        CACHE_FOLDER
        / f"{exp_dir}/nonmate_intermediates.npy"
    )
    gallery_cache.parent.mkdir(exist_ok=True, parents=True)

    if gallery_cache.exists():
        gallery_features = np.load(gallery_cache)
    else:
        for gallery_idx, id_path in enumerate(gallery_id_path):
            gallery_id = id_path.stem

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

    print("Gallery original feature shape: ", gallery_features.shape)
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

    nonmate_ids = os.listdir(nonmate_path)
    nonmate_ids = natsorted(
        filter(lambda x: len(list(os.listdir(nonmate_path / x))) > 0, nonmate_ids)
    )
    if nonmatefeats_nonfused_cache.exists():
        nonmate_nonfused_features = np.load(
            nonmatefeats_nonfused_cache, allow_pickle=True
        )
        nonmate_nonfused_intermediates = np.load(
            nonmateints_nonfused_cache, allow_pickle=True
        )
        assert len(nonmate_nonfused_features) == len(nonmate_nonfused_intermediates)

        nonmate_features = np.zeros((len(nonmate_ids), feature_size))
        for nonmate_idx, (probe_features, probe_intermediates) in enumerate(
            zip(nonmate_nonfused_features, nonmate_nonfused_intermediates)
        ):
            probe_fused_feature, _ = fuse_feature(
                probe_features,
                aggregator,
                probe_intermediates,
                method=fusion_method,
                device=device,
                running_avg_alpha=running_avg_alpha,
            )
            nonmate_features[nonmate_idx] = probe_fused_feature
    else:
        nonmate_features = np.zeros((len(nonmate_ids), feature_size))
        nonmate_nonfused_features = []
        nonmate_nonfused_intermediates = []
        for nonmate_idx, nonmate_id in enumerate(nonmate_ids):
            id_path = nonmate_path / nonmate_id

            nonmate_images = list(id_path.glob("*.[jp][pn]g"))
            dataloader = prepare_imagelist_dataloader(
                nonmate_images, batch_size=16, num_workers=0
            )
            probe_features, probe_intermediates = infer_features(
                dataloader, model, aggregator, hyper_param, device=device
            )
            nonmate_nonfused_features.append(probe_features)
            nonmate_nonfused_intermediates.append(probe_intermediates)
            probe_fused_feature, _ = fuse_feature(
                probe_features,
                aggregator,
                probe_intermediates,
                method=fusion_method,
                device=device,
                running_avg_alpha=running_avg_alpha,
            )
            nonmate_features[nonmate_idx] = probe_fused_feature
        np.save(nonmatefeats_nonfused_cache, nonmate_nonfused_features)
        np.save(nonmateints_nonfused_cache, nonmate_nonfused_intermediates)
        # np.save(nonmate_cache, nonmate_features)

    gallery_sizes = list(sorted(map(lambda x: int(x.strip()), gallery_size.split(","))))
    pbar = tqdm(gallery_sizes)
    break_in_next_loop = False
    for gallery_size in pbar:
        if break_in_next_loop:
            break
        pbar.set_description(f"Eval for gallery size: {gallery_size}")
        if gallery_size > gallery_features.shape[0]:
            print(f"Gallery size {gallery_size} is bigger than total gallery size: {gallery_features.shape}. Evaluate with all gallery feature then quit!")
            gallery_size = gallery_features.shape[0]
            break_in_next_loop = True
        cur_gallery_features = gallery_features[:gallery_size]

        mate_gallery_similarity = np.dot(mate_features, cur_gallery_features.T)
        nonmate_gallery_similarity = np.dot(nonmate_features, cur_gallery_features.T)

        mate_predict_idx, mate_predict_score = np.argmax(
            mate_gallery_similarity, axis=1
        ), np.max(mate_gallery_similarity, axis=1)
        _, nonmate_predict_score = np.argmax(nonmate_gallery_similarity, axis=1), np.max(
            nonmate_gallery_similarity, axis=1
        )
        mate_predict_str = np.array([gallery_ids[idx] for idx in mate_predict_idx])
        mate_label = np.array(
            [mate_id.split("_")[-1].strip().lower() for mate_id in final_mate_folders]
        )

        num_mate_searchs = len(final_mate_folders)
        num_nonmate_searchs = len(nonmate_ids)

        fpirs, fnirs = [], []
        for threshold in np.arange(0, 1, 1 / steps):
            valid_ids = mate_predict_score >= threshold
            valid_predict = mate_predict_str[valid_ids]
            valid_label = mate_label[valid_ids]
            TP = np.sum(valid_predict == valid_label)
            FN = num_mate_searchs - TP

            nonmate_accept_ids = nonmate_predict_score >= threshold
            FP = np.sum(nonmate_accept_ids)

            # Calculate FPIR
            FPIR = FP / num_mate_searchs
            # Calculate FNIR
            FNIR = FN / num_nonmate_searchs
            fpirs.append(FPIR)
            fnirs.append(FNIR)
        np.save(f"{exp_dir}/{gallery_size}_fpir_fnir.npy", {"FNIR": fnirs, "FPIR": fpirs})

@app.command()
def plot_multiple(
    output_dir: Path = typer.Argument(..., help="Output dir"),
    exps: List[Path] = typer.Option([], help="Experiment folders"),
    override_legend: Optional[List[str]] = typer.Option(
        None, "--override", "-o", help="default will be exp folder name"
    ),
):
    if override_legend is None or len(override_legend) == 0:
        override_legend = [exp.name for exp in exps]

    fig1 = plt.figure(figsize=(15, 7))
    ax1 = fig1.add_subplot(111)
    x_labels = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    yticks = np.array(
        [
            0.0,
            0.010,
            0.020,
            0.030,
            0.040,
            0.050,
            0.070,
            0.100,
            0.200,
            0.300,
            0.400,
            0.500,
            0.600,
            0.700,
            0.800,
            0.900,
            1.0,
        ]
    )
    xticks = np.array([0.000001, *x_labels, 1.3])
    xticks_l = ["", "0.0003", "0.001", "0.003", "0.01", "0.03", "0.1", "0.3", "1", ""]
    yticks_l = [
        "",
        "0.010",
        "0.020",
        "0.030",
        "0.040",
        "0.050",
        "0.070",
        "0.100",
        "0.200",
        "0.300",
        "0.400",
        "0.500",
        "0.600",
        "0.700",
        "0.800",
        "0.900",
        "",
    ]

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_yticks(yticks)
    ax1.set_xticks(xticks)
    ax1.xaxis.set_ticklabels(xticks_l)
    ax1.yaxis.set_ticklabels(yticks_l)
    ax1.set_xlim(left=0.0002, right=1.3)
    ax1.set_ylim(bottom=0.08, top=1.0)
    ax1.text(0.002, y=0.1, s="Identification seldom\n uses human review", color="blue")
    ax1.text(0.45, y=0.1, s="Identification always\n uses human review", color="blue")
    ax1.grid(linestyle="--")

    # plt.figure(figsize=(15, 7))
    for exp, legend_name in zip(exps, override_legend):
        fpirs = np.load(exp / "fpir.npy")
        fnirs = np.load(exp / "fnir.npy")

        ax1.plot(fpirs, fnirs, linestyle="solid", label=legend_name)
    # plt.savefig((output_dir / f"det_curve.png").as_posix())

    colormap = plt.cm.Dark2  # nipy_spectral, Set1,Paired
    colors = [colormap(i) for i in np.linspace(0, 0.7, len(ax1.lines))]
    for i, j in enumerate(ax1.lines):
        j.set_color(colors[i])

    ax1.set_title(
        f"DETR Curve",
        fontsize=20,
    )
    ax1.legend()
    ax1.set_xlabel("False Positive Identification Rate", fontsize=16)
    ax1.set_ylabel("False Negative Identification Rate", fontsize=16)
    fig1.savefig((output_dir / f"det_curve.png").as_posix())


if __name__ == "__main__":
    app()
