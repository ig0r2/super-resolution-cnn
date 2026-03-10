from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio, LearnedPerceptualImagePatchSimilarity
from abc import ABC


class Metrics(ABC):
    def __init__(self, metric):
        self.metric = metric
        self.total = 0
        self.n = 0

    def __call__(self, img1, img2):
        return self.metric(img1, img2)  # input must be fp32

    def update(self, img1, img2):
        self.total += self(img1, img2)
        self.n += 1

    def compute(self):
        if self.n == 0: return 0
        return self.total / self.n

    def reset(self):
        self.total = 0
        self.n = 0


class PSNR(Metrics):
    def __init__(self, device, data_range=1.0):
        super().__init__(PeakSignalNoiseRatio(data_range=data_range).to(device))


class SSIM(Metrics):
    def __init__(self, device, data_range=1.0):
        super().__init__(StructuralSimilarityIndexMeasure(data_range=data_range).to(device))


class LPIPS(Metrics):
    def __init__(self, device):
        # normalize converts from [0,1] to [-1,1]
        super().__init__(LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device))
