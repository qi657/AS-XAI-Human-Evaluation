import os

import click
import torch.cuda
from crp.attribution import CondAttribution


from datasets_ import get_dataset, get_sample
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.composites import EpsilonPlusFlat
import zennit.image as zimage

from models import get_model


@click.command()
@click.option("--model_name", default="vgg16")
@click.option("--dataset_name", default="imagenet")
@click.option("--start_sample", default=0, type=int)
@click.option("--end_sample", default=20, type=int)
def crp(model_name, dataset_name,start_sample, end_sample):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(model_name).to(device)
    model.eval()

    canonizers = [SequentialMergeBatchNorm()]
    composite = EpsilonPlusFlat(canonizers)
    dataset = get_dataset(dataset_name)["test"]()

    attribution = CondAttribution(model)

    for sample in range(start_sample, end_sample + 1):

        data, target = get_sample(dataset, sample_id=sample, device=device)
        attr = attribution(data.requires_grad_(), [{"y": target.long().item()}], composite)
        attr = attr.heatmap
        heatmap = zimage.imgify(attr.detach().cpu(), symmetric=True, cmap="bwr")
        os.makedirs(f"results/lrp/{dataset_name}_dog", exist_ok=True)
        heatmap.save(f"results/lrp/{dataset_name}_dog/sample_{sample}_{model_name}_lrp.png")


if __name__ == "__main__":
    crp()
