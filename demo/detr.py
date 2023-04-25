import typer
import matplotlib.pyplot as plt
import torch
import cv2
from natsort import natsorted
from prettytable import PrettyTable
import os
from pathlib import Path
import numpy as np
import sys

sys.path.append("/home/termanteus/workspace/face/code/caface/demo")
sys.path.append("/home/termanteus/workspace/face/code/caface")

CACHE_FOLDER = Path("./cache")
CACHE_FOLDER.mkdir(parents=True, exist_ok=True)

from dataset import prepare_imagelist_dataloader, to_tensor
from face_detection.detector import FaceDetector
from face_alignment.aligner import FaceAligner
from model_loader import load_caface
from inference import infer_features, fuse_feature, l2_normalize
import visualization

app = typer.Typer()


@app.command()
def main(
    gallery_path: Path = typer.Argument(..., help="Path to gallery images"),
    mate_path: Path = typer.Argument(..., help="Path to mate images"),
    nonmate_path: Path = typer.Argument(..., help="Path to nonmate images"),
    ckpt_path: Path = typer.Argument(..., help="Path to checkpoint"),
    output_dir: Path = typer.Argument(..., help="Path to ouput"),
    feature_size: int = typer.Option(512, help="Feature size"),
    steps: int = typer.Option(1000, help="threshold step"),
    device: str = typer.Option("cuda:0", help="Device"),
    # 'cluster_and_aggregate', 'norm_weight', 'average'
    fusion_method: str = typer.Option("average", help="fusion method"),
):
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

    gallery_cache = CACHE_FOLDER / f"{ckpt_path.stem}_{gallery_path.stem}.npy"
    mate_cache = CACHE_FOLDER / f"{ckpt_path.stem}_{mate_path.stem}_{fusion_method}.npy"
    nonmate_cache = (
        CACHE_FOLDER / f"{ckpt_path.stem}_{nonmate_path.stem}_{fusion_method}.npy"
    )

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
    if mate_cache.exists():
        mate_features = np.load(mate_cache)
    else:
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
            probe_fused_feature, _ = fuse_feature(
                probe_features,
                aggregator,
                probe_intermediates,
                method=fusion_method,
                device=device,
            )
            mate_features[mate_idx] = probe_fused_feature
        np.save(mate_cache, mate_features)

    nonmate_ids = os.listdir(nonmate_path)
    nonmate_ids = natsorted(
        filter(lambda x: len(list(os.listdir(nonmate_path / x))) > 0, nonmate_ids)
    )
    if nonmate_cache.exists():
        nonmate_features = np.load(nonmate_cache)
    else:
        nonmate_features = np.zeros((len(nonmate_ids), feature_size))
        for nonmate_idx, nonmate_id in enumerate(nonmate_ids):
            id_path = nonmate_path / nonmate_id

            nonmate_images = list(id_path.glob("*.[jp][pn]g"))
            dataloader = prepare_imagelist_dataloader(
                nonmate_images, batch_size=16, num_workers=0
            )
            probe_features, probe_intermediates = infer_features(
                dataloader, model, aggregator, hyper_param, device=device
            )
            probe_fused_feature, _ = fuse_feature(
                probe_features,
                aggregator,
                probe_intermediates,
                method=fusion_method,
                device=device,
            )
            nonmate_features[nonmate_idx] = probe_fused_feature
        np.save(nonmate_cache, nonmate_features)

    mate_gallery_similarity = np.dot(mate_features, gallery_features.T)
    nonmate_gallery_similarity = np.dot(nonmate_features, gallery_features.T)

    mate_predict_idx, mate_predict_score = np.argmax(
        mate_gallery_similarity, axis=1
    ), np.max(mate_gallery_similarity, axis=1)
    _, nonmate_predict_score = np.argmax(nonmate_gallery_similarity, axis=1), np.max(
        nonmate_gallery_similarity, axis=1
    )
    mate_predict_str = np.array([gallery_ids[idx] for idx in mate_predict_idx])
    mate_label = np.array([mate_id.split("_")[-1].strip().lower() for mate_id in final_mate_folders])

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


if __name__ == "__main__":
    app()
