from torchvision import datasets, transforms
from visualize_utils import *

# Quick script to analyse original datasets

if __name__ == "__main__":
    # Create transforms object to put image to tensor
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Load datasets
    train_dataset = datasets.ImageFolder("dataset/train", transform=transform)
    test_dataset = datasets.ImageFolder("dataset/test", transform=transform)
    classes = train_dataset.classes

    # Size of sets
    print(f"Training set size: {len(train_dataset)}")
    print(f"Testing set size: {len(test_dataset)}")

    # Visualize some examples
    visualize_dataset(train_dataset, classes, name="train")
    visualize_dataset(test_dataset, classes, name="test")

    # See proportions of each classes in training set
    visualize_class_balance(train_dataset)
