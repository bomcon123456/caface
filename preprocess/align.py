import cv2
import numpy as np
from skimage import transform as trans
import typer
from pathlib import Path
from tqdm.rich import tqdm

arcface_src = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

arcface_src = np.expand_dims(arcface_src, axis=0)

# lmk is prediction; src is template
def estimate_norm(lmk, image_size=112, mode="arcface"):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float("inf")

    assert image_size == 112
    src = arcface_src
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))

        if error < min_error:
            min_error = error
            min_M = M
            min_index = i

    return min_M, min_index


def norm_crop(img, landmark, image_size=112, mode="arcface"):
    M, _ = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped

app = typer.Typer()

@app.command()
def main(
    ijb_loosecrop_path: Path = typer.Argument(..., help="path to image root"),
    ijb_txt: Path = typer.Argument(..., help="path to image list"),
    output_path: Path = typer.Argument(..., help="path to outpath"),
    dataset_name: str = typer.Option("IJBB", help="dataset name"),
):
    landmark_list_path = ijb_txt
    img_list = open(landmark_list_path)
    files = img_list.readlines()
    print('Total files:', len(files))
    output_path.mkdir(exist_ok=True, parents=True)

    # img_paths = []
    for each_line in tqdm(files):
        name_lmk_score = each_line.strip().split(' ')
        lmks = np.array(name_lmk_score[1:11], dtype=np.float32).reshape(-1, 2)
        img_path = ijb_loosecrop_path / name_lmk_score[0]
        # img_paths.append(img_path)
        img = cv2.imread(img_path.as_posix())
        aligned = norm_crop(img, lmks, image_size=112, mode="arcface")
        outpath = output_path / name_lmk_score[0]
        cv2.imwrite(outpath.as_posix(), aligned)

if __name__ == "__main__":
    app()