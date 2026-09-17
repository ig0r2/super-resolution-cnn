import torch


# Pretvara iz OpenCV formata u format za model, odradi inference i onda vrati u format za OpenCV
class VideoWrapperCV2(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[..., [2, 1, 0]] / 255.0  # BGR [0-255] -> RGB [0-1]
        x = x.permute(2, 0, 1).unsqueeze(0)  # (H,W,C) -> (1,C,H,W)
        out = self.model(x)
        out = torch.clamp(out, 0.0, 1.0) * 255.0
        # (1,C,H,W) -> (H,W,C) and RGB -> BGR
        out = out.squeeze(0).permute(1, 2, 0)[..., [2, 1, 0]]
        return out.to(dtype=torch.uint8)
