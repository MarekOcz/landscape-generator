from torch.cuda import is_available, current_device
from torch.version import cuda
from torch import __version__
from matplotlib.pyplot import subplots
from numpy import ndarray, asarray
from typing import Iterable, Sized
from PIL.Image import open as img_open
from pathlib import Path
import torch


def diagnostics():
    print("Pytorch version:", __version__)
    if is_available():
        print(f"CUDA is available. CUDA version: {cuda}\n"
              f"ID of current CUDA device: {current_device()}\n"
              "===================================\n")
        torch.backends.cudnn.benchmark = True
        return torch.device(f"cuda:{current_device()}")
    else:
        print("CUDA is not available\n"
              "===================================\n")
        return torch.device("cpu")


def show_pic_grid(length: int, height: int, images: Iterable[ndarray] | Sized, titles: Iterable | Sized = None):
    """
    Shows a grid of inputted pictures
    :param length: How many rows
    :param height: How many columns
    :param images: A list of images.
    :param titles: A list of titles for the inputted images. Leave as None if there isn't any titles
    :return: figure and axes for pyplot.
    """
    # Input sanitization
    if titles is None:
        titles = [''] * len(images)
    if (length * height) != len(images):
        raise IndexError(
            f"There are {len(images)} images inputted which does not match the {length * height} spaces in the grid")
    if len(titles) != len(images):
        raise IndexError(
            f"There are {len(images)} images but {len(titles)} titles. These need to be the same to work")

    # Main
    fig, axs = subplots(height, length)
    for i, row in enumerate(axs):
        for j, column in enumerate(row):
            column.imshow(images[i * length + j])
            column.set_title(titles[i * length + j])
            column.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
            column.xaxis.set_visible(False)
            column.yaxis.set_visible(False)
    return fig, axs


def load_images_with_titles(paths: list[Path]) -> tuple[list[ndarray], list[str]]:
    """

    :param paths: A list of pathlib Paths to the images
    :return:  A list of images as numpy arrays and their corresponding titles in another list
    """
    imgs = []
    titles = []
    for path in paths:
        titles.append(str(path).split("\\")[-2])
        with img_open(path) as img:
            # noinspection PyTypeChecker
            imgs.append(asarray(img))
    return imgs, titles


if __name__ == '__main__':
    pass
