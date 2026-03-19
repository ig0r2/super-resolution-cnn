import torch


def tile_forward(model, upscale_factor, img, tile_size=256, overlap=32):
    """
    Inferenca tako sto se slika podeli u vise delova (tiles) koji se na kraju spoje

    """
    b, c, h, w = img.shape
    if b != 1:
        raise ValueError("tile_predict batch size must be 1")

    scale = upscale_factor
    step = tile_size - overlap

    hs, ws = h * scale, w * scale

    output = torch.zeros((1, c, hs, ws), device=img.device, dtype=img.dtype)
    count = torch.zeros((1, 1, hs, ws), device=img.device, dtype=torch.float32)

    for y in range(0, h, step):
        y2 = min(y + tile_size, h)
        y1 = max(0, y2 - tile_size)
        ys1, ys2 = y1 * scale, y2 * scale

        for x in range(0, w, step):
            x2 = min(x + tile_size, w)
            x1 = max(0, x2 - tile_size)
            xs1, xs2 = x1 * scale, x2 * scale

            tile = img[:, :, y1:y2, x1:x2]  # [1,C,th,tw]
            pred = model(tile)  # [1,C,th*s,tw*s]

            output[:, :, ys1:ys2, xs1:xs2] += pred
            count[:, :, ys1:ys2, xs1:xs2] += 1.0

    return output / count.clamp_min(1.0).to(dtype=output.dtype)
