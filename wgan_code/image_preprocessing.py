from PIL import Image, UnidentifiedImageError
from PIL.Image import Resampling
from os import listdir, makedirs
from os.path import splitext, isdir
from typing import IO, Iterable
from math import floor
from random import seed, shuffle
from shutil import copy2
from tqdm.auto import tqdm


def resize_all(from_path: IO | str, to_path: IO | str, size: tuple[int, int]):
    """
    Resizes all images in the from_path directory or in branches of it and stores them in the to_path directory with the
    same directory structure as the from_path. Run recursively so may throw an error if run with deep directories

    :param from_path: The directory where the images are originally stored
    :param size: The shape of the images to be resized to
    :param to_path: Where the resized images should be stored
    """
    # noinspection PyTypeChecker
    for file in tqdm(listdir(from_path), ncols=200):
        try:
            with Image.open(f"{from_path}/{file}") as img:
                img = img.resize(size, Resampling.LANCZOS)
                img.save(fp=f"{to_path}/{splitext(file)[0] + '.png'}",  # Changes the filename to be .png
                         format="png")
        except (PermissionError, UnidentifiedImageError):
            try:
                listdir(f"{from_path}/{file}")
                makedirs(f"{to_path}/{file}", exist_ok=True)
                resize_all(f"{from_path}/{file}", f"{to_path}/{file}", size)  # Recursively runs for inner directories
            except (NotADirectoryError, FileNotFoundError):
                pass
    print(f"Completed {from_path}")


def convert_to_rgb(from_path: IO | str, to_path: IO | str):
    """
    Resizes all images in the from_path directory or in branches of it and stores them in the to_path directory with the
    same directory structure as the from_path. Run recursively so may throw an error if run with deep directories

    :param from_path: The directory where the images are originally stored
    :param to_path: Where the resized images should be stored
    """
    # noinspection PyTypeChecker
    for file in tqdm(listdir(from_path), ncols=100):
        try:
            with Image.open(f"{from_path}/{file}") as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(fp=f"{to_path}/{splitext(file)[0] + '.png'}",  # Changes the filename to be .png
                         format="png")
        except (PermissionError, UnidentifiedImageError):
            try:
                listdir(f"{from_path}/{file}")
                makedirs(f"{to_path}/{file}", exist_ok=True)
                convert_to_rgb(f"{from_path}/{file}", f"{to_path}/{file}")  # Recursively runs for inner directories
            except (NotADirectoryError, FileNotFoundError):
                pass
    print(f"Completed {from_path}")


def train_test_file_split(from_path: IO | str, train_path: IO | str, test_path: IO | str,
                          file_extensions: Iterable[str] = None, test_split: float = 0.2):
    """
    Splits a directory in a stratified way into a train and test directory to be ready for loading with pytorch. Run
    recursively so may throw an error if run with very deep directories
    :param from_path: The directory where the files are originally stored
    :param train_path: The directory where the training files are to be copied to e.g. 'root/data/train'
    :param test_path: The directory where the testing files are to be copied to e.g. 'root/data/test'
    :param file_extensions: The file extensions to be included. Do not include the '.' before the value e.g. 'jpg' rather than '.jpg'. Leave as None to include all files
    :param test_split: The split of files to be moved to the test directory
    :return:
    """
    makedirs(train_path, exist_ok=True)
    makedirs(test_path, exist_ok=True)

    if floor(test_split) != 0 or test_split == 0:
        raise ValueError(f"test_split must be a value between but not including, 0 or 1. You inputted {test_split}")
    files_to_split = []
    for file in listdir(from_path):
        if isdir(f"{from_path}/{file}"):
            train_test_file_split(f"{from_path}/{file}", f"{train_path}/{file}",
                                  f"{test_path}/{file}", file_extensions, test_split)
        elif file_extensions is None:
            files_to_split.append(file)
        elif splitext(file)[1][1:] in file_extensions:
            files_to_split.append(file)
    shuffle(files_to_split)
    if len(files_to_split) > 0:
        test_num = round(test_split * len(files_to_split))  # Calculate the number of files
        for test_file in files_to_split[:test_num]:  # Iterates through the files in the test split
            copy2(src=f"{from_path}/{test_file}", dst=test_path)  # Copies the files
        for train_file in files_to_split[test_num:]:
            copy2(src=f"{from_path}/{train_file}", dst=train_path)


if __name__ == '__main__':
    pass
