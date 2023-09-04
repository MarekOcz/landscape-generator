import math
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import DataLoader
from pathlib import Path


def load_dataset(folder: str | Path, batch_size: int = 64, img_channels: int = 3, device: str = 'cpu',
                 num_workers: int = 0, pin_memory: bool = False) -> DataLoader:
    """
    Loads the dataset with normalised images of 0.5 mean and standard deviation as tensors
    :param folder:  The folder where the data is stored. Ensure the data is inside another folder to mimic the labels
    ImageFolder class to work e.g. root/dataset/images/1.png, 2.png, ...
    :param batch_size: The batch size of the loaded data
    :param img_channels: The number of image channels
    :param device: The device to load the data onto. Must be a string and not a torch.device object
    :param num_workers: The number of cores to load the data onto
    :param pin_memory: If True, the data loader will copy Tensors into device/CUDA pinned memory before returning them.
    (copied from torch.utils.data.DataLoader docs).
    :return: An iterable dataloader containing all the images
    """
    transforms = Compose([
        ToTensor(),
        Normalize([0.5] * img_channels,
                  [0.5] * img_channels),
    ])

    dataset = ImageFolder(folder, transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory,
                      pin_memory_device=str(device) if pin_memory else "", num_workers=num_workers)


if __name__ == '__main__':
    pass
