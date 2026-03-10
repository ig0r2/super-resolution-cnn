from typing import Literal
import torch
from PIL import Image
from torch import nn
from torchvision.transforms import v2, InterpolationMode

METHODS = {
    'nearest': InterpolationMode.NEAREST,
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
}


class RegularModel(nn.Module):
    def __init__(self, name: Literal['nearest', 'bilinear', 'bicubic', 'lanczos'], upscale_factor):
        """
        torch Module wrapper za klasicne metode interpolacije
        """
        super().__init__()

        if name not in METHODS.keys():
            raise ValueError(f"Method {name} not available. Available methods: {list(METHODS.keys())}")

        self.method = METHODS[name]
        self.upscale_factor = upscale_factor

    def forward(self, x):
        # lanczos nije podrzan na GPU, mora na CPU
        if self.method == InterpolationMode.LANCZOS:
            if x.shape[0] > 1:
                raise ValueError(f"Only supports batch of 1")
            img = v2.functional.to_pil_image(x.squeeze(0).cpu())

            new_size = [x.shape[-1] * self.upscale_factor, x.shape[-2] * self.upscale_factor]
            img = img.resize(new_size, resample=Image.LANCZOS)

            img = v2.functional.to_image(img)
            img = v2.functional.to_dtype(img, dtype=torch.float, scale=True).unsqueeze(0).to(x.device)
            return img

        # resize sa torchvision koristeci trazeni metod
        new_size = [x.shape[-2] * self.upscale_factor, x.shape[-1] * self.upscale_factor]
        return v2.functional.resize_image(x, new_size, interpolation=self.method, antialias=True)
