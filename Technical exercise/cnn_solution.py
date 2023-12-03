import os
import torch
import argparse
import random
import numpy as np
from torchvision import transforms, datasets
from cnn_trainer import Trainer
from visualize_utils import *
from data_utils import *


def main(args):
    """
    Main fonction to process data, train and evaluate our classifier.
    Parameters:
        args: Arguments fed to command-line interface (or default values).
    """

    # 1 - Preprocess data :
    # - Load datasets
    # - Split training set into training and validation set
    # - Use a "conservative" transformation on validation and testing set
    # - Create two copies of training set:
    #       One with a "conservative" transformation, one with
    #       modifications to perform data augmentation
    # - Concatenate the two copies to obtained an augmented training set

    conservative_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    augmentation_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(30),
            transforms.RandomGrayscale(p=0.1),
            transforms.ColorJitter(brightness=0.2, hue=0.1),
            transforms.RandomInvert(p=0.2),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.1),
            transforms.RandomErasing(p=0.1),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    full_train_dataset = datasets.ImageFolder("dataset/train")
    test_dataset = datasets.ImageFolder(
        "dataset/test", transform=conservative_transform
    )
    classes = full_train_dataset.classes

    tmp_train_dataset, val_dataset = train_val_split(full_train_dataset, args.seed)
    val_dataset = SubsetToDataset(val_dataset, transform=conservative_transform)
    original_train_dataset = SubsetToDataset(
        tmp_train_dataset, transform=conservative_transform
    )
    augmented_train_dataset = SubsetToDataset(
        tmp_train_dataset, transform=augmentation_transform
    )
    train_dataset = torch.utils.data.ConcatDataset(
        [original_train_dataset, augmented_train_dataset]
    )

    # 2 - Moderate class imbalance
    # Calculate weights depending on each class proportion
    # for each image in the augmented training set.
    # Those weights will be used by an WeightedRandomSampler
    # object which will oversample the minority class when
    # creating batches.

    targets = []
    for _, target in train_dataset:
        targets.append(target)

    class_sample_count = [(targets == t).sum() for t in np.unique(np.sort(targets))]
    weights = [1 / c for c in class_sample_count]
    sample_weights = torch.tensor([weights[t] for t in targets])

    # 3 - Declare the Trainer object

    trainer = Trainer(
        args,
        classes,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        sampling_weights=sample_weights,
    )

    # 5 - Train model and visualize training and validation
    # losses, save the model.
    # If the model was already saved, it is only loaded
    # for evaluation.

    if not os.path.exists(args.filename) or args.force:
        print("--- Training model ---\n")
        tr_loss, val_loss = trainer.train()
        trainer.save()
        visualize_loss(tr_loss, val_loss, filename=args.filename)
    else:
        print(f"--- Loading model  from {args.filename} file ---\n")
        trainer.load()

    # 6 - Evaluate model performance on all sets

    train_results, train_errors, _ = trainer.eval(mode="train")
    val_results, val_errors, _ = trainer.eval(mode="val")
    test_results, test_errors, test_preds = trainer.eval(mode="test")

    # 7 - Display classification report results
    # and misclassified images

    print(f"\n--- Results on training set ---\n\n{train_results}")
    visualize_errors(
        train_dataset, classes, train_errors, name="train", filename=args.filename
    )

    print(f"\n--- Results on validation set ---\n\n{val_results}")
    visualize_errors(
        val_dataset, classes, val_errors, name="val", filename=args.filename
    )

    print(f"\n--- Results on testing set ---\n\n{test_results}")
    visualize_errors(
        test_dataset, classes, test_errors, name="test", filename=args.filename
    )

    # 8 - Visualize inference on testing set
    visualize_inference(test_dataset, classes, test_preds, name="test", filename=args.filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for random initialization"
    )
    parser.add_argument(
        "--image_size", default=224, type=int, help="Dimensions for resizing images."
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size for training and evaluating.",
    )
    parser.add_argument(
        "--lr", default=3e-4, type=float, help="Learning rate for Adam optimizer."
    )
    parser.add_argument(
        "--n_epochs", default=40, type=int, help="Number of training epochs."
    )
    parser.add_argument(
        "--filename", default="classifier", type=str, help="Filename to save model."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="To force model training if --filename already exists.",
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    main(args)
