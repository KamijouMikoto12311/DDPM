import os
import time

import cv2
import einops
import numpy as np
import torch
import torch.nn as nn

from dataset import get_dataloader
from ddpm import DDPM
from Unet import Unet

from accelerate import Accelerator


def train(ddpm: DDPM, net, config: dict, accelerator: Accelerator, ckpt_path="model.pth"):
    print(f"config:{config}")
    batch_size = config["batch_size"]
    n_epochs = config["n_epochs"]

    n_steps = ddpm.n_steps

    dataloader = get_dataloader(batch_size)
    # net = net.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), 1e-3)

    dataloader, net, optimizer = accelerator.prepare(dataloader, net, optimizer)

    tic = time.time()
    for e in range(n_epochs):
        total_loss = 0

        for x, _ in dataloader:
            current_batch_size = x.shape[0]
            # x = x.to(device)
            t = torch.randint(0, n_steps, (current_batch_size,)).to(x.device)
            eps = torch.randn_like(x).to(x.device)
            x_t = ddpm.sample_forward(x, t, eps)
            eps_theta = net(x_t, t)
            loss = loss_fn(eps_theta, eps)
            optimizer.zero_grad()
            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            total_loss += loss.item() * current_batch_size

        total_loss /= len(dataloader.dataset)
        toc = time.time()

        # torch.save(net.state_dict(), ckpt_path)
        if accelerator.is_main_process:
            print(f"epoch {e} loss: {total_loss} elapsed {(toc - tic):.2f}s")
            raw_net = accelerator.unwrap_model(net)
            accelerator.save(raw_net.state_dict(), ckpt_path)

    print("Done")


if __name__ == "__main__":
    accelerator = Accelerator()
    os.makedirs("work_dirs", exist_ok=True)

    batch_size = 1024
    n_epochs = 100

    config = {"batch_size": batch_size, "n_epochs": n_epochs}

    n_steps = 1000
    device = accelerator.device
    model_path = "model_unet.pth"

    net = Unet(dim=10)
    ddpm = DDPM(device, n_steps)

    train(ddpm, net, config, accelerator=accelerator, ckpt_path=model_path)
