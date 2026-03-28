from pathlib import Path
from typing import Literal

import torch
from torchvision.io import read_image
from torchvision.utils import save_image

from models import RegularModel
from utils.checkpoints import load_model_from_checkpoint
from utils.model_utils import tile_forward

# Inferenca svih slika iz inference/input foldera za izabrani checkpoint ili metod

CHECKPOINT_PATH: Path = Path("checkpoints/SR_RFDN_2x_2_64.pth")

USE_METHOD = False
METHOD: Literal['nearest', 'bilinear', 'bicubic', 'lanczos'] = "bicubic"
UPSCALE_FACTOR = 2

INPUT_DIR: Path = Path("inference/input")
OUTPUT_DIR: Path = Path("inference/output")

#####################################################

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)
if USE_METHOD:
    print("Using method", METHOD)
else:
    print("Using checkpoint", CHECKPOINT_PATH.stem)

# Load model
if USE_METHOD:
    model = RegularModel(METHOD, UPSCALE_FACTOR)
else:
    model, _ = load_model_from_checkpoint(CHECKPOINT_PATH, device)

model.eval()

for input_path in INPUT_DIR.iterdir():
    if not input_path.is_file(): continue
    input_tensor = read_image(str(input_path)).unsqueeze(0).float().div_(255.0).to(device)

    with torch.no_grad():
        # output = model(input_tensor)
        output = tile_forward(model, UPSCALE_FACTOR, input_tensor)

    output = output.squeeze().cpu().clamp(0, 1)

    # save output image
    if USE_METHOD:
        output_image_path = OUTPUT_DIR / f"{input_path.stem}_{METHOD}.png"
    else:
        output_image_path = OUTPUT_DIR / f"{input_path.stem}_{CHECKPOINT_PATH.stem}.png"
    save_image(output, output_image_path)

    print(
        f"Processed: {input_path.name} ({input_tensor.shape[3]}x{input_tensor.shape[2]} -> {output.shape[2]}x{output.shape[1]})")

print(f"Finished! All images saved to: {OUTPUT_DIR}")
