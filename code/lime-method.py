# -- coding: utf-8 --
# @Time : 2024/2/1 17:07
# @Author : Harper
# @Email : sunc696@gmail.com
# @File : lime-method.py
import os
import click
from lime import lime_image
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as TF

from models import get_model
from datasets_ import get_dataset, get_sample
import matplotlib.pyplot as plt


@click.command()
@click.option("--model_name", default="vgg16")
@click.option("--dataset_name", default="imagenet")
@click.option("--start_sample", default=0, type=int)
@click.option("--end_sample", default=20, type=int)
def lime_explain(model_name, dataset_name, start_sample, end_sample):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(model_name).to(device)
    model.eval()

    for sample in range(start_sample, end_sample + 1):
        dataset = get_dataset(dataset_name)["test"]()
        data, target = get_sample(dataset, sample_id=sample, device=device)

        def batch_predict(images):
            model.eval()
            batch = torch.stack(tuple(TF.to_tensor(i) for i in images), dim=0)

            device = next(model.parameters()).device
            batch = batch.to(device)

            logits = model(batch)
            probs = torch.nn.functional.softmax(logits, dim=1)
            return probs.detach().cpu().numpy()

        explainer = lime_image.LimeImageExplainer()

        explanation = explainer.explain_instance(data[0].cpu().numpy().transpose(1, 2, 0),
                                                 batch_predict, top_labels=5, hide_color=0,
                                                 num_samples=1000)  # Use 1000 randomly perturbed samples

        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10,
                                                    hide_rest=False)
        original_image = data[0].cpu().numpy().transpose(1, 2, 0)

        mask_expanded = np.expand_dims(mask, axis=2)
        mask_expanded = np.repeat(mask_expanded, 3, axis=2)

        masked_image = mask_expanded * original_image
        heatmap = np.sum(masked_image, axis=2)
        heatmap_image = Image.fromarray(np.uint8(plt.cm.viridis(heatmap) * 255))


        os.makedirs(f"results/lime/{dataset_name}", exist_ok=True)
        heatmap_image.save(f"results/lime/{dataset_name}/sample_{sample}_{model_name}_lime.png")


if __name__ == "__main__":
    lime_explain()
