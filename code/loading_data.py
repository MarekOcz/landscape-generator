import math
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import DataLoader
from torch.cuda import is_available
from utils import diagnostics
from tqdm.auto import tqdm


def load_dataset(folder: str, batch_size: int = 64, img_channels: int = 3, device='cpu', num_workers: int = 0,
                 pin_memory: bool = False):
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
