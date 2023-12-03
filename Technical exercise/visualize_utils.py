import matplotlib.pyplot as plt
import math
import numpy as np
from collections import Counter


def visualize_dataset(dataset, classes, name, num_images=10, num_rows=2):
    """
    Function to visualize RGB images data inside a pytorch Dataset.
    Parameters:
        dataset: the dataset object to visualize.
        classes: classes names for visualization.
        num_images: number of images to display.
        num_rows: number of row to display the image on.
    """
    num_cols = math.ceil(num_images / 2)
    fig, axs = plt.subplots(num_rows, num_cols)
    for i in range(num_rows):
        for j in range(num_cols):
            image, label = dataset[np.random.randint(0, len(dataset))]
            image = (image + 1) / 2
            axs[i, j].imshow(image.permute(1, 2, 0).squeeze())
            axs[i, j].axis("off")
            axs[i, j].set_title(
                f"{classes[label]}",
                fontsize=12,
                color="green" if classes[label] == classes[0] else "orange",
            )
    fig.suptitle("Data examples")
    plt.savefig(f"{name}_examples.png")
    plt.show()


def visualize_class_balance(dataset, name="train"):
    """
    Function to visualize RGB images data inside a pytorch Dataset.
    Parameters:
        dataset: the dataset object to visualize.
        name: name of dataset visualized to save image.
    """
    class_counts = Counter(dataset.targets)
    plt.bar(dataset.classes, class_counts.values(), color=["green", "orange"])
    plt.title("Class balance")
    plt.savefig(f"{name}_balance.png")
    plt.show()


def visualize_loss(training_loss, validation_loss, filename="exp"):
    """
    Function to visualize progression of losses through epochs.
    Parameters:
        training_loss: the training loss to visualize in a dictionnary.
        validation_loss: the validation loss to visualize in a dictionnary.
        filename: name of experiment to save image.
    """
    fig, ax = plt.subplots()
    ax.plot(
        list(training_loss.keys()),
        list(training_loss.values()),
        color="blue",
        label="training",
    )
    if len(validation_loss) != 0:
        ax.plot(
            list(validation_loss.keys()),
            list(validation_loss.values()),
            color="orange",
            label="validation",
        )
    ax.set_title("Losses evolution through training")
    ax.set_xticks(np.arange(0, len(training_loss), 5))
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    plt.savefig(f"{filename}_loss_evolution.png")
    plt.show()


def visualize_errors(dataset, classes, errors_indices, name="train", filename="exp"):
    """
    Function to visualize images uncorrectly classified.
    Parameters:
        dataset: the dataset object to visualize.
        classes: classes names for visualization.
        errors_indices: indexes/indices where classification erros occured.
        name: name of dataset visualized to save image.
        filename: name of experiment to save image.
    """
    if len(errors_indices) == 0:
        return

    num_rows = math.ceil(len(errors_indices) / 3)
    num_cols = 3
    fig, axs = plt.subplots(num_rows, num_cols, squeeze=False)
    num = 0
    for i in range(num_rows):
        for j in range(num_cols):
            axs[i, j].axis("off")
            if num < len(errors_indices):
                image, label = dataset[errors_indices[num]]
                image = (image + 1) / 2
                axs[i, j].imshow(image.permute(1, 2, 0).squeeze())
                axs[i, j].set_title(
                    f"Should be: {classes[label]}",
                    fontsize=12,
                    color="green" if classes[label] == classes[0] else "orange",
                )
                num += 1
    fig.suptitle("Data uncorrectly classified")
    plt.savefig(f"{filename}_{name}_errors.png")
    plt.show()

def visualize_inference(dataset, classes, predictions, name="test", filename="exp"):
    """
    Function to visualize inference over dataset.
    Parameters:
        dataset: the dataset object to visualize.
        classes: classes names for visualization.
        preds: indexes/indices where classification erros occured.
        name: name of dataset visualized to save image.
        filename: name of experiment to save image.
    """

    num_rows = math.ceil(len(predictions) / 4)
    num_cols = 4
    fig, axs = plt.subplots(num_rows, num_cols, squeeze=False)
    num = 0
    for i in range(num_rows):
        for j in range(num_cols):
            axs[i, j].axis("off")
            if num < len(predictions):
                image, _ = dataset[num]
                image = (image + 1) / 2
                axs[i, j].imshow(image.permute(1, 2, 0).squeeze())
                axs[i, j].set_title(
                    f"{classes[predictions[num]]}",
                    fontsize=12,
                    color="green" if classes[predictions[num]] == classes[0] else "orange",
                )
                num += 1
    fig.suptitle("Data inference")
    plt.savefig(f"{filename}_{name}_inference.png")
    plt.show()
