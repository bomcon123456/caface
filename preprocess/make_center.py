import sys
import pyrootutils
import os
import typer
from pathlib import Path

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
sys.path.append(os.path.dirname(root))
sys.path.append(os.path.join(os.path.dirname(root), 'caface'))

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from caface import model as model_module
import torch
from mxdataset import MXDataset
from backbones import get_model


def groupby_ops(value: torch.Tensor, labels: torch.LongTensor, op='sum') -> (torch.Tensor, torch.LongTensor):
    uniques = labels.unique().tolist()
    labels = labels.tolist()

    key_val = {key: val for key, val in zip(uniques, range(len(uniques)))}
    val_key = {val: key for key, val in zip(uniques, range(len(uniques)))}

    labels = torch.LongTensor(list(map(key_val.get, labels)))

    labels = labels.view(labels.size(0), 1).expand(-1, value.size(1))

    unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
    result = torch.zeros_like(unique_labels, dtype=value.dtype).scatter_add_(0, labels, value)
    if op == 'mean':
        result = result / labels_count.float().unsqueeze(1)
    else:
        assert op == 'sum'
    new_labels = torch.LongTensor(list(map(val_key.get, unique_labels[:, 0].tolist())))
    return result, new_labels, labels_count


app = typer.Typer()

@app.command()
def main(
    rec_path: Path = typer.Argument(..., help="path to image root"),
    backbone_name: str = typer.Argument(..., help="backbone name"),
    model_path: Path = typer.Argument(..., help="model path"),
    save_dir: Path = typer.Argument(..., help="save dir path"),
):
    name = "center_{}_{}.pth".format(
        backbone_name, model_path.parent.name)
    save_dir.mkdir(exist_ok=True, parents=True)
    print('saving at')
    print(save_dir / name)

    # load model (This model assumes the input to be BGR image (cv2), not RGB (pil))
    net = get_model(backbone_name, fp16=False)
    net.load_state_dict(torch.load(model_path))
    net.to("cuda:0")
    net.eval()

    with torch.no_grad():
        batch_size = 256
        train_dataset = MXDataset(root_dir=rec_path.as_posix())
        dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
        center = torch.zeros((len(train_dataset.record_info.label.unique()), 512))
        cul_count = torch.zeros(len(train_dataset.record_info.label.unique()))
        for batch in tqdm(dataloader):
            img, tgt = batch
            embedding, norm = net(img.cuda())
            sum_embedding, new_tgt, labels_count = groupby_ops(embedding.detach().cpu(), tgt, op='sum')
            for emb, tgt, count in zip(sum_embedding, new_tgt, labels_count):
                center[tgt] += emb
                cul_count[tgt] += count

        # flipped version
        train_dataset = MXDataset(root_dir=rec_path.as_posix(), flip_probability=1.0)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
        for batch in tqdm(dataloader):
            img, tgt = batch
            embedding, norm = net(img.cuda())
            sum_embedding, new_tgt, labels_count = groupby_ops(embedding.detach().cpu(), tgt, op='sum')
            for emb, tgt, count in zip(sum_embedding, new_tgt, labels_count):
                center[tgt] += emb
                cul_count[tgt] += count

    # normalize
    center = center / cul_count.unsqueeze(-1)
    center = center / torch.norm(center, 2, -1, keepdim=True)

    torch.save({'center': center, 'model': model_path.as_posix(), 'dataset': rec_path.as_posix()},
               (save_dir/name).as_posix())

if __name__ == "__main__":
    app()
