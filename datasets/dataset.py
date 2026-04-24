import random

import torch
from tqdm import tqdm
import sys
import torch.utils.data as data
import torch.nn.functional as F
from torchvision.io import decode_image
from torchvision.transforms import v2

from .transforms import RandomRotation90
from .dataset_utils import jpeg_quality_for_index, apply_jpeg_compression, sharpen_image, ensure_rgb, crop_to_match


# Trening dataset klasa
# cuva nazive fajlova HR slika (filenames)
# tokom treniranja primenjuje random crop (velicine patch_size), random flip, random rotation pa tek onda napravi i LR sliku
# preload = False - uvek ucitava slike sa diska
# preload = True - ucita sve slike u RAM pre nego sto pocne treniranje. Za DIV2K potrebno je 10GB slobodno
# Uz ovu klasu mora da se koristi TrainCollateFn kao collate_fn u dataloaderu
class ImageDatasetTrain(data.Dataset):
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
            self.hr_images = [decode_image(str(p)) for p in
                              tqdm(self.filenames, desc="Loading images", unit="img", file=sys.stdout)]

    def __getitem__(self, index):
        # format slike od decode_image je tensor RGB [C, H, W] uint8
        hr = self.hr_images[index] if self.preload else decode_image(str(self.filenames[index]))
        # transforimisi
        return self.transforms(hr)

    def __len__(self):
        return len(self.filenames)


class TrainCollateFn:
    def __init__(self, scale, jpeg_degradation=False, jpeg_quality=(20, 95)):
        self.scale = scale
        self.jpeg_degradation = jpeg_degradation
        self.jpeg_quality = jpeg_quality  # (min, max) quality range

    def __call__(self, batch):
        """The collate_fn in DataLoader is a function that processes a list of samples to create a batch.

        Stacks the batch of HR images, then it downscales whole batch. Slightly faster than downscaling in __getitem__"""
        hr = torch.stack(batch)
        _, _, H, W = hr.shape
        if self.scale > 1:
            lr = F.interpolate(hr, size=(H // self.scale, W // self.scale), mode='bicubic', align_corners=False,
                               antialias=True)
        else:
            lr = hr.clone()

        hr = hr.float().div(255.0)

        # JPEG degradation on LR
        if self.jpeg_degradation:
            lr = apply_jpeg_compression(lr, random.randint(*self.jpeg_quality))
            hr = sharpen_image(hr)  # sharpen HR

        # float (0-1)
        return lr.float().div(255.0), hr


# Test dataset klasa
# cuva filenames od HR i LR slika [(LR,HR)]
# normalize = True - pretvara slike u float [0-1]
# normalize = False - ostavlja slike kao unit8 [0-255]
class ImageDatasetTest(data.Dataset):
    def __init__(self, filenames, upscale_factor, preload=False, normalize=True, jpeg_degradation=False):
        super().__init__()
        self.filenames = filenames
        self.preload = preload
        self.upscale_factor = upscale_factor
        self.normalize = normalize
        self.jpeg_degradation = jpeg_degradation
        # preload
        if preload:
            self.hr_images = []
            self.lr_images = []
            for i, (path) in enumerate(tqdm(filenames, desc="Loading images", unit="img", file=sys.stdout)):
                lr = decode_image(str(path[0]))
                hr = decode_image(str(path[1]))
                lr, hr = self._transform(lr, hr, i)
                self.lr_images.append(lr)
                self.hr_images.append(hr)

    def _transform(self, lr, hr, index):
        # neke slike su grayscale tj imaju samo jedan kanal
        lr = ensure_rgb(lr)
        hr = ensure_rgb(hr)
        # neke slike nisu deljive faktorom uvecanja
        target_h = hr.shape[1] - hr.shape[1] % self.upscale_factor
        target_w = hr.shape[2] - hr.shape[2] % self.upscale_factor
        hr = crop_to_match(hr, target_h, target_w)

        # 100 images, quality 20-100
        if self.jpeg_degradation:
            lr = apply_jpeg_compression(lr, jpeg_quality_for_index(index))

        return lr, hr

    def __getitem__(self, index):
        if self.preload:
            lr = self.lr_images[index]
            hr = self.hr_images[index]
        else:
            lr = decode_image(str(self.filenames[index][0]))
            hr = decode_image(str(self.filenames[index][1]))
            lr, hr = self._transform(lr, hr, index)

        if self.normalize:
            return lr.float().div(255.0), hr.float().div(255.0)
        return lr, hr

    def __len__(self):
        return len(self.filenames)
