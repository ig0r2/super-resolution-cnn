import shutil
import urllib.request
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from .dataset import ImageDatasetTrain, ImageDatasetTest
from .dataset_multiscale import ImageDatasetMultiscaleTrain, ImageDatasetMultiscaleTest


def check_if_dataset_exists(path: Path):
    return path.is_dir() and any(path.iterdir())


def download_dataset(url, path: Path):
    path.mkdir(parents=True, exist_ok=True)

    # Progress callback function
    def reporthook(block_num, block_size, total_size):
        if total_size > 0:
            percent = min(block_num * block_size * 100 / total_size, 100)
            filled = int(30 * percent / 100)
            print(f'\r|{"█" * filled}{" " * (30 - filled)}| {percent:.1f}%', end='', flush=True)

    # Download the file to a temporary location
    print(f"Downloading from {url}")
    temp_file, _ = urllib.request.urlretrieve(url, reporthook=reporthook)
    print()

    filename = Path(urlparse(url).path).name
    archive_path = path / filename
    shutil.move(temp_file, str(archive_path))

    try:
        print(f"Extracting {filename}")
        shutil.unpack_archive(str(archive_path), str(path))
        print("Extraction successful")
    except shutil.ReadError:
        print(f"Error: {filename} is not a supported format")
    finally:
        # Clean up the archive file after extraction
        if archive_path.exists(): archive_path.unlink()


# glavna funkcija za uzimanje div2k training seta
def get_training_set(upscale_factor, patch_size, preload, data_dir="./data"):
    data_dir = Path(data_dir) / 'DIV2K'
    hr_dir = data_dir / 'DIV2K_train_HR'
    if not check_if_dataset_exists(hr_dir):
        download_dataset("http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip", data_dir)

    filenames = sorted([f for f in hr_dir.iterdir() if f.is_file()])

    print(f"Using training set: DIV2K {upscale_factor}x")
    return ImageDatasetTrain(filenames=filenames, upscale_factor=upscale_factor, patch_size=patch_size, preload=preload)


# Za multiscale trening
def get_training_set_multi(patch_size, preload, data_dir="./data"):
    data_dir = Path(data_dir) / 'DIV2K'
    hr_dir = data_dir / 'DIV2K_train_HR'
    if not check_if_dataset_exists(hr_dir):
        download_dataset("http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip", data_dir)

    filenames = sorted([f for f in hr_dir.iterdir() if f.is_file()])

    print(f"Using training set: DIV2K")
    return ImageDatasetMultiscaleTrain(filenames=filenames, patch_size=patch_size, preload=preload)


def combine_filenames(lr_dir: Path, hr_dir: Path):
    input_files = sorted([f for f in lr_dir.iterdir() if f.is_file()])
    target_files = sorted([f for f in hr_dir.iterdir() if f.is_file()])
    return [(a, b) for a, b in zip(input_files, target_files)]


def get_div2k_test_set(upscale_factor: Literal[2, 3, 4], preload, normalize, data_dir="./data"):
    if upscale_factor not in [2, 3, 4]:
        raise Exception(f'Upscale Factor {upscale_factor} unsupported in dataset')

    data_dir = Path(data_dir) / 'DIV2K'
    lr_dir = data_dir / 'DIV2K_valid_LR_bicubic' / f"X{upscale_factor}"
    hr_dir = data_dir / 'DIV2K_valid_HR'

    if not check_if_dataset_exists(hr_dir):
        download_dataset("http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip", data_dir)

    if not check_if_dataset_exists(lr_dir):
        download_dataset(f"http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X{upscale_factor}.zip",
                         data_dir)

    print(f"Using test set: DIV2K {upscale_factor}x")
    return ImageDatasetTest(filenames=combine_filenames(lr_dir, hr_dir), upscale_factor=upscale_factor, preload=preload,
                            normalize=normalize)


# test/validation set that contains 2x,3x,4x LR
def get_div2k_test_set_multi(preload, normalize, data_dir="./data"):
    data_dir = Path(data_dir) / 'DIV2K'
    lr_dir_2 = data_dir / 'DIV2K_valid_LR_bicubic/X2'
    lr_dir_3 = data_dir / 'DIV2K_valid_LR_bicubic/X3'
    lr_dir_4 = data_dir / 'DIV2K_valid_LR_bicubic/X4'
    hr_dir = data_dir / 'DIV2K_valid_HR'

    if not check_if_dataset_exists(hr_dir):
        download_dataset("http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip", data_dir)

    if not check_if_dataset_exists(lr_dir_2):
        download_dataset(f"http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X{2}.zip",
                         data_dir)
    if not check_if_dataset_exists(lr_dir_3):
        download_dataset(f"http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X{3}.zip",
                         data_dir)
    if not check_if_dataset_exists(lr_dir_4):
        download_dataset(f"http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X{4}.zip",
                         data_dir)

    lr_files_2 = sorted([f for f in lr_dir_2.iterdir() if f.is_file()])
    lr_files_3 = sorted([f for f in lr_dir_3.iterdir() if f.is_file()])
    lr_files_4 = sorted([f for f in lr_dir_4.iterdir() if f.is_file()])
    target_files = sorted([f for f in hr_dir.iterdir() if f.is_file()])
    filenames = [(a, b, c, d) for a, b, c, d in zip(lr_files_2, lr_files_3, lr_files_4, target_files)]

    print(f"Using test set: DIV2K 2x,3x,4x")
    return ImageDatasetMultiscaleTest(filenames=filenames, preload=preload, normalize=normalize)


def get_hugginface_test_set(name, upscale_factor, preload, normalize, data_dir="./data"):
    if upscale_factor not in [2, 3, 4]:
        raise Exception(f'Upscale Factor {upscale_factor} unsupported in dataset')

    data_dir = Path(data_dir) / f'{name}'
    hr_dir = data_dir / f'{name}_HR'
    lr_dir = data_dir / f'{name}_LR_x{upscale_factor}'

    if not check_if_dataset_exists(hr_dir):
        download_dataset(
            f"https://huggingface.co/datasets/eugenesiow/{name}/resolve/main/data/{name}_HR.tar.gz?download=true",
            data_dir)

    if not check_if_dataset_exists(lr_dir):
        download_dataset(
            f"https://huggingface.co/datasets/eugenesiow/{name}/resolve/main/data/{name}_LR_x{upscale_factor}.tar.gz?download=true",
            data_dir)

    print(f"Using test set {name} {upscale_factor}x")
    return ImageDatasetTest(filenames=combine_filenames(lr_dir, hr_dir), upscale_factor=upscale_factor, preload=preload,
                            normalize=normalize)


# glavna funkcija za uzimanje test seta
def get_test_set(name: Literal["DIV2K", "Set5", "Set14", "BSD100", "Urban100"],
                 upscale_factor: Literal[2, 3, 4], preload, normalize=True, data_dir="./data"):
    if name.upper() == "DIV2K":
        return get_div2k_test_set(upscale_factor, preload, normalize, data_dir)
    if name.upper() == "SET5":
        return get_hugginface_test_set("Set5", upscale_factor, preload, normalize, data_dir)
    if name.upper() == "SET14":
        return get_hugginface_test_set("Set14", upscale_factor, preload, normalize, data_dir)
    if name.upper() == "BSD100":
        return get_hugginface_test_set("BSD100", upscale_factor, preload, normalize, data_dir)
    if name.upper() == "URBAN100":
        return get_hugginface_test_set("Urban100", upscale_factor, preload, normalize, data_dir)
    raise ValueError(f"Test set {name} is not available. Available test sets: DIV2K, Set5, Set14, BSD100, Urban100")
