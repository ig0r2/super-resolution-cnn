import random
import sys

import torch
import torch.utils.data as data
import torch.nn.functional as F
from torchvision.io import decode_image
from torchvision.transforms import v2
from tqdm import tqdm

from .transforms import RandomRotation90


class ImageDatasetMultiscaleTrain(data.Dataset):
    def __init__(self, filenames, patch_size=192, preload=False):
        super().__init__()

        self.filenames = filenames
        self.preload = preload

        # transforms
        self.transforms = v2.Compose([
            v2.RandomCrop((patch_size, patch_size)),
            v2.RandomHorizontalFlip(),
            RandomRotation90()
        ])
        # preload
        if preload:
            self.hr_images = []
            for path in tqdm(filenames, desc="Loading images", unit="img", file=sys.stdout):
                self.hr_images.append(decode_image(str(path)))

    def __getitem__(self, index):
        # format slike od read_image je tensor RGB [C, H, W] uint8
        if self.preload:
            hr = self.hr_images[index]
        else:
            hr = decode_image(str(self.filenames[index]))
        # transforimisi
        hr = self.transforms(hr)
        # uint8 (0-255) -> float (0-1)
        return hr.float().div(255.0)

    def __len__(self):
        return len(self.filenames)


def multiscale_train_collate_fn(batch):
    """The collate_fn in DataLoader is a function that processes a list of samples to create a batch.

    Stacks the batch of HR images, then it randomly chooses a scale and downscales whole batch"""
    hr = torch.stack(batch)
    scale = random.choice([2, 3, 4])

    _, _, H, W = hr.shape
    lr = F.interpolate(hr, size=(H // scale, W // scale), mode='bicubic', align_corners=False, antialias=True)

    return lr, hr, scale


# cuva filenames od HR i LR slika [(LR2x,LR3x,LR4x,HR)] - saves RAM and a bit of time compared to 3 separate val loaders
class ImageDatasetMultiscaleTest(data.Dataset):
    def __init__(self, filenames, preload=False, normalize=True):
        super().__init__()
        self.filenames = filenames
        self.preload = preload
        self.normalize = normalize
        # preload
        if preload:
            self.hr_images = []
            self.lr_images_2 = []
            self.lr_images_3 = []
            self.lr_images_4 = []
            for path in tqdm(filenames, desc="Loading images", unit="img", file=sys.stdout):
                lr_2 = decode_image(str(path[0]))
                lr_3 = decode_image(str(path[1]))
                lr_4 = decode_image(str(path[2]))
                hr = decode_image(str(path[3]))
                lr_2, lr_3, lr_4, hr = self._transform(lr_2, lr_3, lr_4, hr)
                self.lr_images_2.append(lr_2)
                self.lr_images_3.append(lr_3)
                self.lr_images_4.append(lr_4)
                self.hr_images.append(hr)

    def _crop_to_match(self, img, target_h, target_w):
        diff_h = img.shape[1] - target_h
        diff_w = img.shape[2] - target_w
        if diff_h == 0 and diff_w == 0:
            return img
        return v2.functional.crop_image(img, diff_h // 2, diff_w // 2, target_h, target_w)

    def _transform(self, lr_2, lr_3, lr_4, hr):
        # neke slike su grayscale tj imaju samo jedan kanal
        if lr_2.shape[0] != 3: lr_2 = v2.functional.grayscale_to_rgb(lr_2)
        if lr_3.shape[0] != 3: lr_3 = v2.functional.grayscale_to_rgb(lr_3)
        if lr_4.shape[0] != 3: lr_4 = v2.functional.grayscale_to_rgb(lr_4)
        if hr.shape[0] != 3: hr = v2.functional.grayscale_to_rgb(hr)

        # crop HR to be divisible by 12
        target_h = hr.shape[1] - hr.shape[1] % 12
        target_w = hr.shape[2] - hr.shape[2] % 12
        hr = self._crop_to_match(hr, target_h, target_w)

        # derive exact LR target sizes from HR
        lr_2 = self._crop_to_match(lr_2, target_h // 2, target_w // 2)
        lr_3 = self._crop_to_match(lr_3, target_h // 3, target_w // 3)
        lr_4 = self._crop_to_match(lr_4, target_h // 4, target_w // 4)

        return lr_2, lr_3, lr_4, hr

    def __getitem__(self, index):
        if self.preload:
            lr_2 = self.lr_images_2[index]
            lr_3 = self.lr_images_3[index]
            lr_4 = self.lr_images_4[index]
            hr = self.hr_images[index]
        else:
            lr_2 = decode_image(str(self.filenames[index][0]))
            lr_3 = decode_image(str(self.filenames[index][1]))
            lr_4 = decode_image(str(self.filenames[index][2]))
            hr = decode_image(str(self.filenames[index][3]))
            lr_2, lr_3, lr_4, hr = self._transform(lr_2, lr_3, lr_4, hr)

        if self.normalize:
            return lr_2.float().div(255.0), lr_3.float().div(255.0), lr_4.float().div(255.0), hr.float().div(255.0)
        return lr_2, lr_3, lr_4, hr

    def __len__(self):
        return len(self.filenames)
