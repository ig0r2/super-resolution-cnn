import matplotlib.pyplot as plt
from utils.path import get_logs_path


def smooth_curve(points, factor=0.8):
    """
    Applies Exponential Moving Average (EMA) for smoothing.
    A factor closer to 1.0 means more smoothing.
    """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def plot_training_history(history, title, save_path=None, smooth_factor=0.8):
    plt.style.use('seaborn-v0_8-whitegrid')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('white')

    # Plot 1: Loss
    ax1 = axes[0]
    train_epochs = [d['epoch'] for d in history['training']]
    train_loss = [d['loss'] for d in history['training']]
    train_loss_smoothed = smooth_curve(train_loss, factor=smooth_factor)

    # Plot raw loss faintly in the background
    ax1.plot(train_epochs, train_loss, linewidth=1.0, alpha=0.2, color='blue')
    # Plot smoothed loss boldly on top
    ax1.plot(train_epochs, train_loss_smoothed, linewidth=2.0, label=title, alpha=0.9, color='blue')

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

    ax2.plot(val_epochs, val_ssim, linewidth=1.5, marker='D', markersize=3, label=title, alpha=0.9, color='orange')

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('SSIM', fontsize=12)
    ax2.set_title('Validation SSIM', fontsize=14)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim([0.7, 1])

    plt.tight_layout()

    if save_path is None:
        save_path = get_logs_path(f'training/{title}.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to: {save_path}")

    plt.show()


def plot_training_history_gan(history, title, save_path=None, smooth_factor=0.8):
    plt.style.use('seaborn-v0_8-whitegrid')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('white')

    # Plot 1: GAN Losses (Discriminator & Generator)
    ax1 = axes[0]
    train_epochs = [d['epoch'] for d in history['training']]

    train_d_loss = [d['d_loss'] for d in history['training']]
    train_g_loss = [d['g_loss'] for d in history['training']]

    # Smooth both losses
    train_d_loss_smoothed = smooth_curve(train_d_loss, factor=smooth_factor)
    train_g_loss_smoothed = smooth_curve(train_g_loss, factor=smooth_factor)

    # Plot raw D loss faintly
    ax1.plot(train_epochs, train_d_loss, linewidth=1.0, alpha=0.2, color='blue')
    # Plot smoothed D loss
    ax1.plot(train_epochs, train_d_loss_smoothed, linewidth=2.0,
             label='Discriminator Loss', alpha=0.9, color='blue')

    # Plot raw G loss faintly
    ax1.plot(train_epochs, train_g_loss, linewidth=1.0, alpha=0.2, color='green')
    # Plot smoothed G loss
    ax1.plot(train_epochs, train_g_loss_smoothed, linewidth=2.0,
             label='Generator Loss', alpha=0.9, color='green')

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Losses (D & G)', fontsize=14)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Plot 2: Validation
    ax2 = axes[1]
    val_epochs = [d['epoch'] for d in history['validation']]
    val_score = [d['score'] for d in history['validation']]

    ax2.plot(val_epochs, val_score, linewidth=1.5,
             marker='D', markersize=3, label=title, alpha=0.9, color='orange')

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Validation Score', fontsize=14)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    # ax2.set_ylim([0.5, 1])

    plt.tight_layout()

    if save_path is None:
        save_path = get_logs_path(f'training/{title}.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to: {save_path}")

    plt.show()
