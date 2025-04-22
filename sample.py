import torch
import einops
import cv2
import numpy as np

from Unet import Unet
from ddpm import DDPM
from dataset import get_img_shape


def sample_imgs(ddpm, net, output_path, n_sample=81, device="cuda"):
    net = net.to(device)
    net = net.eval()
    with torch.no_grad():
        shape = (n_sample, *get_img_shape())
        imgs = ddpm.sample_backward(shape, net, device=device).detach().cpu()
        imgs = (imgs + 1) / 2 * 255
        imgs = imgs.clamp(0, 255)
        imgs = einops.rearrange(imgs, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(n_sample**0.5))

        imgs = imgs.numpy().astype(np.uint8)
        imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)

        cv2.imwrite(output_path, imgs)


if __name__ == "__main__":
    device = "cuda"
    n_steps = 1000
    model_path = "model_unet.pth"

    net = Unet(dim=10)
    ddpm = DDPM(device, n_steps)
    net.load_state_dict(torch.load(model_path))
    sample_imgs(ddpm, net, "work_dirs/diffusion.jpg", device=device)
