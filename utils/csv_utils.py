import csv
from pathlib import Path


def save_to_csv(data, csv_path):
    """
    Save or append evaluation data to a CSV file.
    :param data: Dictionary with evaluation metrics
    :param csv_path: Path to the CSV file
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())

        # Write header only if file is new
        if not file_exists:
            writer.writeheader()

        writer.writerow(data)

    print(f"Results saved to {csv_path}")
