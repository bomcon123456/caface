import sys
from pathlib import Path
import pyrootutils
import os
import torch

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
sys.path.append(os.path.dirname(root))
sys.path.append("/home/termanteus/workspace/face/code/caface/demo")

import argparse
import cv2
from face_detection.detector import FaceDetector
from face_alignment.aligner import FaceAligner
from model_loader import load_caface
from dataset import get_all_files, natural_sort, prepare_imagelist_dataloader, to_tensor
from tqdm import tqdm
import numpy as np
from inference import infer_features, fuse_feature, l2_normalize
import visualization

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("probe_path", type=str)
    parser.add_argument("gallery_path", type=str)
    parser.add_argument("--ckpt_path", type=str, default='../pretrained_models/CAFace_AdaFaceWebFace4M.ckpt')
    parser.add_argument("--save_root", type=str, default='result/examples1')
    parser.add_argument("--device", type=str, default='cuda:0')
    # parser.add_argument("--fusion_method", type=str,
    #                     default='cluster_and_aggregate',
    #                     choices=['cluster_and_aggregate', 'norm_weight', 'average'])

    args = parser.parse_args()

    # load face detector and aligner
    detector = FaceDetector()
    aligner = FaceAligner()

    # load caface
    aggregator, model, hyper_param = load_caface(args.ckpt_path, device=args.device)

    # make save_dir
    save_dir = os.path.join(args.save_root, os.path.basename(args.probe_path))
    os.makedirs(save_dir, exist_ok=True)

    probe_image_list = list(map(lambda x: x.as_posix(),Path(args.probe_path).rglob("*.jpg")))
    dataloader = prepare_imagelist_dataloader(probe_image_list, batch_size=16, num_workers=0)

    # infer singe image features
    probe_features, probe_intermediates = infer_features(dataloader, model, aggregator, hyper_param, device=args.device)
    # fuse features
    probe_feats, probe_ws = [], []
    fusion_methods = ['cluster_and_aggregate', 'norm_weight', 'average']
    for fusion_method in fusion_methods:
        probe_fused_feature, probe_weights = fuse_feature(probe_features, aggregator, probe_intermediates,
                                                        method=fusion_method, device=args.device)
        probe_feats.append(probe_fused_feature)
        probe_ws.append(probe_weights)

    # infer gallery for comparison with probe video
    gallery_path = args.gallery_path
    if os.path.isfile(gallery_path):
        # infer gallery feature
        gallery_image = aligner.align(detector.detect(cv2.imread(gallery_path)))
        gallery_image_tensor = to_tensor(gallery_image, device=args.device)
        with torch.no_grad():
            gallery_feature, _ = model(gallery_image_tensor)
        gallery_feature = gallery_feature.detach().cpu().numpy()
        gallery_feature = l2_normalize(gallery_feature)
    if os.path.isdir(gallery_path):
        # infer gallery feature
        p = Path(gallery_path)
        gallery_images = p.glob("*.[jp][pn]g")
        fs = []
        for gallery_path in gallery_images:
            gallery_image = aligner.align(detector.detect(cv2.imread(gallery_path.as_posix())))
            gallery_image_tensor = to_tensor(gallery_image, device=args.device)
            with torch.no_grad():
                gallery_feature, _ = model(gallery_image_tensor)
            gallery_feature = gallery_feature.detach().cpu().numpy()
            fs.append(gallery_feature)
        gallery_feature = l2_normalize(np.array(fs).mean(0))

    # make cosine similarity plot
    for fusion_idx, fusion_method in enumerate(fusion_methods):
        visualization.make_similarity_plot_multiple(os.path.join(save_dir, f'{fusion_method}.pdf'),
                                        probe_features, probe_ws, probe_feats, probe_image_list,
                                        gallery_feature, gallery_image, fusion_methods, fusion_idx)
