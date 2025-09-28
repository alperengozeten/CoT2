import matplotlib.pyplot as plt


def plot_losses(epochs_range, train_losses, val_losses, model_type: str, output_dir: str = ''):
    plt.figure(figsize=(8,6))
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses,   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'figures/{output_dir}{model_type}_model_train_val_loss.pdf', format='pdf', bbox_inches='tight')


def plot_accuracies(epochs_range, val_accuracies, model_type: str, output_dir: str = ''):
    plt.figure(figsize=(8,6))
    plt.plot(epochs_range, val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.title('Validation Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'figures/{output_dir}{model_type}_model_val_accuracy.pdf', format='pdf', bbox_inches='tight')
