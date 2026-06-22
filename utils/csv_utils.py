from pathlib import Path

import pandas as pd


def save_to_csv(data, csv_path):
    """
    Update or append evaluation data to a CSV file.
    :param data: Dictionary with evaluation metrics
    :param csv_path: Path to the CSV file
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df_new = pd.DataFrame([data])

    if csv_path.exists():
        df = pd.read_csv(csv_path, dtype=str)
        mask = df['model_name'] == data['model_name']
        if mask.any():
            for key, value in data.items():
                if value != "" and key in df.columns:
                    df.loc[mask, key] = str(value)
        else:
            df = pd.concat([df, df_new.astype(str)], ignore_index=True)
    else:
        df = df_new.astype(str)

    df.to_csv(csv_path, index=False)


def get_columns_to_evaluate(csv_path, model_name):
    metric_cols = {'LPIPS', 'SSIM', 'PSNR', 'Loss'}
    perf_720p_cols = {'FPS 720p', 'VRAM (MB) 720p'}
    perf_480p_cols = {'FPS 480p', 'VRAM (MB) 480p'}

    # If no CSV yet, evaluate everything that's requested
    if not Path(csv_path).exists():
        return True, True, True

    df = pd.read_csv(csv_path)
    row = df[df['model_name'] == model_name]

    # Not in CSV at all, evaluate everything requested
    if row.empty:
        return True, True, True

    row = row.iloc[0]

    # Check which parts are missing/empty
    needs_metrics = row[list(metric_cols)].isnull().any()
    needs_720p = row[list(perf_720p_cols)].isnull().any()
    needs_480p = row[list(perf_480p_cols)].isnull().any()

    return needs_metrics, needs_720p, needs_480p
