import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda, ToTensor


def download_dataset():
    cifar100 = torchvision.datasets.CIFAR100(root="./data/cifar100", download=True)
    print("length of CIFAR100", len(cifar100))


def get_dataloader(batch_size: int):
    transform = Compose([ToTensor(), Lambda(lambda x: (x - 0.5) * 2)])
    dataset = torchvision.datasets.CIFAR100(root="./data/cifar100", transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_img_shape():
    return (3, 32, 32)


if __name__ == "__main__":
    import os

    os.makedirs("work_dirs", exist_ok=True)
    download_dataset()
