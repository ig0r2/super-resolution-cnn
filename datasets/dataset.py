import torch
from tqdm import tqdm
import sys
import torch.utils.data as data
import torch.nn.functional as F
from torchvision.io import decode_image
from torchvision.transforms import v2

from .transforms import RandomRotation90


# Trening dataset klasa
# cuva nazive fajlova HR slika (filenames)
# tokom treniranja primenjuje random crop (velicine patch_size), random flip, random rotation pa tek onda napravi i LR sliku
# preload = False - uvek ucitava slike sa diska
# preload = True - ucita sve slike u RAM pre nego sto pocne treniranje. Za DIV2K potrebno je 10GB slobodno
# Uz ovu klasu mora da se koristi TrainCollateFn kao collate_fn u dataloaderu
class ImageDatasetTrain(data.Dataset):
    def __init__(self, filenames, upscale_factor, patch_size=192, preload=False):
        super().__init__()
        self.filenames = filenames
        self.preload = preload
        self.upscale_factor = upscale_factor

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
        # format slike od decode_image je tensor RGB [C, H, W] uint8
        if self.preload:
            hr = self.hr_images[index]
        else:
            hr = decode_image(str(self.filenames[index]))

        # transforimisi
        hr = self.transforms(hr)
        return hr

    def __len__(self):
        return len(self.filenames)


class TrainCollateFn:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, batch):
        """The collate_fn in DataLoader is a function that processes a list of samples to create a batch.

        Stacks the batch of HR images, then it downscales whole batch. Slightly faster than downscaling in __getitem__"""
        hr = torch.stack(batch)
        _, _, H, W = hr.shape
        lr = F.interpolate(hr, size=(H // self.scale, W // self.scale), mode='bicubic', align_corners=False,
                           antialias=True)
        # uint8 (0-255) -> float (0-1)
        return lr.float().div(255.0), hr.float().div(255.0)


# Test dataset klasa
# cuva filenames od HR i LR slika [(LR,HR)]
# normalize = True - pretvara slike u float [0-1]
# normalize = False - ostavlja slike kao unit8 [0-255]
class ImageDatasetTest(data.Dataset):
    def __init__(self, filenames, upscale_factor, preload=False, normalize=True):
        super().__init__()
        self.filenames = filenames
        self.preload = preload
        self.upscale_factor = upscale_factor
        self.normalize = normalize
        # preload
        if preload:
            self.hr_images = []
            self.lr_images = []
            for path in tqdm(filenames, desc="Loading images", unit="img", file=sys.stdout):
                lr = decode_image(str(path[0]))
                hr = decode_image(str(path[1]))
                lr, hr = self._transform(lr, hr)
                self.lr_images.append(lr)
                self.hr_images.append(hr)

    def _transform(self, lr, hr):
        # neke slike su grayscale tj imaju samo jedan kanal
        if lr.shape[0] != 3: lr = v2.functional.grayscale_to_rgb(lr)
        if hr.shape[0] != 3: hr = v2.functional.grayscale_to_rgb(hr)
        # neke slike nisu deljive faktorom uvecanja
        if hr.shape[1] % self.upscale_factor != 0 or hr.shape[2] % self.upscale_factor != 0:
            diff_h = hr.shape[1] % self.upscale_factor
            diff_w = hr.shape[2] % self.upscale_factor
            hr = v2.functional.crop_image(hr, diff_h // 2, diff_w // 2, hr.shape[1] - diff_h, hr.shape[2] - diff_w)
        return lr, hr

    def __getitem__(self, index):
        if self.preload:
            lr = self.lr_images[index]
            hr = self.hr_images[index]
        else:
            lr = decode_image(str(self.filenames[index][0]))
            hr = decode_image(str(self.filenames[index][1]))
            lr, hr = self._transform(lr, hr)

        if self.normalize:
            return lr.float().div(255.0), hr.float().div(255.0)
        return lr, hr

    def __len__(self):
        return len(self.filenames)
