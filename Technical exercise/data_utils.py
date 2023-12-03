from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Subset


class SubsetToDataset(Dataset):
    """
    Class to perform different transforms on subsets originally from the same Dataset.
    Parameters:
        subset: part of a splitted dataset.
        transform: transformation to perform.
    """

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def train_val_split(dataset, seed, val_size=0.1):
    """
    Function to split a dataset into a two subsets, using sklearn train_test_split function.
    Parameters:
        dataset: dataset to split.
        val_size: value of the val split size, to pass to the train_test_split function.
    """
    train_idx, validation_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=val_size,
        shuffle=True,
        stratify=dataset.targets,
        random_state=seed,
    )
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, validation_idx)
    return train_dataset, val_dataset
