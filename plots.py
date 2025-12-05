import matplotlib
import matplotlib.pyplot as plt 

from train import results


matplotlib.use('QtAgg')


def plot_train_results(
        train_losses: list,
        test_losses: list,
        train_accuracy: list,
        test_accuracy: list,
        epochs: int = 10,
    ) -> None:

    _, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    plt.title('Model train results')

    axes[0].plot([*range(epochs)], train_losses, label='train loss')
    axes[0].plot([*range(epochs)], test_losses, label='test loss')
    axes[1].plot([*range(epochs)], train_accuracy, label='train accuracy')
    axes[1].plot([*range(epochs)], test_accuracy, label='test_accuracy')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_train_results(
        train_losses=results['train_loss'],
        test_losses=results['test_loss'],
        train_accuracy=results['train_acc'],
        test_accuracy=results['test_acc'],
        epochs=15
    )