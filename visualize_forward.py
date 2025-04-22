import cv2
import einops
import torch
import numpy as np

from dataset import get_dataloader
from ddpm import DDPM


def visualize_forward():

    n_steps = 200
    device = "cuda"
    dataloader = get_dataloader(5)
    x, _ = next(iter(dataloader))
    x = x.to(device)

    ddpm = DDPM(device, n_steps)
    xts = []
    percents = torch.linspace(0, 0.99, 20)
    for percent in percents:
        t = torch.tensor([int(n_steps * percent)])
        t = t.unsqueeze(1)
        x_t = ddpm.sample_forward(x, t)
        xts.append(x_t)

    res = torch.stack(xts, 0)
    res = einops.rearrange(res, "n1 n2 c h w -> (n2 h) (n1 w) c")
    res = (res.clip(-1, 1) + 1) / 2 * 255
    res = res.cpu().numpy().astype(np.uint8)

    cv2.imwrite("work_dirs/diffusion_forward.jpg", res)


if __name__ == "__main__":
    visualize_forward()
