from pathlib import Path
import matplotlib.pyplot as plt
import torch


def plot_training_history(history, title, save_path=None):
    plt.style.use('seaborn-v0_8-whitegrid')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('white')

    # Plot 1: Loss
    ax1 = axes[0]
    train_epochs = [d['epoch'] for d in history['training']]
    train_loss = [d['loss'] for d in history['training']]

    ax1.plot(train_epochs, train_loss, linewidth=1.2,
             marker='o', markersize=2, label=title, alpha=0.9)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim([0, 0.04])

    # Plot 2: Validation SSIM
    ax2 = axes[1]
    val_epochs = [d['epoch'] for d in history['validation']]
    val_ssim = [d['ssim'] for d in history['validation']]

    ax2.plot(val_epochs, val_ssim, linewidth=1.5,
             marker='D', markersize=3, label=title, alpha=0.9)

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('SSIM', fontsize=12)
    ax2.set_title('Validation SSIM', fontsize=14)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim([0.7, 1])

    plt.tight_layout()

    if save_path is None:
        save_path = Path('logs/training') / f'{title}.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to: {save_path}")

    plt.show()
