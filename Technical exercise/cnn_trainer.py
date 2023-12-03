import os
import torch
import numpy as np
from torch import nn
from tqdm import tqdm, trange
from torch.utils.data import (
    DataLoader,
    SequentialSampler,
    RandomSampler,
    WeightedRandomSampler,
)
from sklearn.metrics import classification_report
from cnn_classifier import Classifier


class Trainer(object):
    """
    Trainer object to train, evaluate, save and load classifier.
    Parameters:
        args: Arguments passed with argparse objects (or default values).
        classes: List of classe's names ('roads' and 'fiels').
        train_dataset: Training set Dataset object.
        val_dataset: Validation set Dataset object.
        test_dataset: Testing set Dataset object.
        sampling_weights: Weight to take into consideration training set
            imbalance using WeightedRandomSampler. Optionnal.

    """

    def __init__(
        self,
        args,
        classes,
        train_dataset=None,
        val_dataset=None,
        test_dataset=None,
        sampling_weights=None,
    ):
        self.args = args
        self.classes = classes
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Classifier().to(self.device)
        self.sample_weights = sampling_weights

    def train(self):
        """
        Perform training on train_dataset for args.n_epochs.
        Use a Adam optimizer, a ReduceLROnPlateau learning rate
            scheduler and a CrossEntropy loss.
        The scheduler use validation loss and if no validation
            set was given to trainer object, no scheduling happen.
        Returns:
            Tuple of training loss history
                and validation loss history over epochs.
        """
        if self.train_dataset == None:
            raise Exception("No training set was given to Trainer object !")

        if self.sample_weights == None:
            train_sampler = RandomSampler(self.train_dataset)
        else:
            train_sampler = WeightedRandomSampler(
                self.sample_weights, len(self.sample_weights), replacement=True
            )
        train_dataloader = DataLoader(
            self.train_dataset, sampler=train_sampler, batch_size=self.args.batch_size
        )

        if self.val_dataset != None:
            val_sampler = SequentialSampler(self.val_dataset)
            val_dataloader = DataLoader(
                self.val_dataset, sampler=val_sampler, batch_size=self.args.batch_size
            )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        criterion = nn.CrossEntropyLoss()

        tr_loss_history = {}
        val_loss_history = {}

        train_iterator = trange(int(self.args.n_epochs), desc="Epoch")

        for epoch in train_iterator:
            cur_step = 0
            cur_loss = 0
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                logits = self.model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                cur_step += 1
                cur_loss += loss.item()

            tr_loss_history[str(epoch + 1)] = cur_loss / cur_step
            cur_step = 0
            cur_loss = 0

            if self.val_dataset != None:
                self.model.eval()
                cur_vstep = 0
                cur_vloss = 0
                with torch.no_grad():
                    for step, vbatch in enumerate(val_dataloader):
                        vimages, vlabels = vbatch
                        vimages = vimages.to(self.device)
                        vlabels = vlabels.to(self.device)
                        vlogits = self.model(vimages)
                        vloss = criterion(vlogits, vlabels)
                        cur_vstep += 1
                        cur_vloss += vloss.item()
                    last_vloss = cur_vloss / cur_vstep
                    val_loss_history[str(epoch + 1)] = last_vloss
                self.model.train()
                scheduler.step(last_vloss)

        return tr_loss_history, val_loss_history

    def eval(self, mode="train"):
        """
        Perform evaluation on either, train/val/test set.
        Parameters:
            mode: Either "train"/"val"/"test"
                for dataset to perform evaluation on.
        Returns :
            scikit-learn classification report object and
                indices of classification errors occures.
        """
        if mode == "train":
            dataset = self.train_dataset
        elif mode == "test":
            dataset = self.test_dataset
        elif mode == "val":
            dataset = self.val_dataset
        else:
            raise Exception(
                "Mode is not recognized (must be either 'train'/'val'/'test')"
            )

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(
            dataset, sampler=eval_sampler, batch_size=self.args.batch_size
        )

        preds = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                images, labels = batch
                logits = self.model(images)
            if preds is None:
                preds = logits.detach().cpu().numpy()
                trues = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                trues = np.append(trues, labels.detach().cpu().numpy(), axis=0)

        preds = np.argmax(preds, axis=1)

        class_report = classification_report(trues, preds, target_names=self.classes)
        errors = np.where(trues != preds)

        return class_report, errors[0], preds

    def save(self):
        """
        Save model under args.filename path.
        """
        if os.path.dirname(self.args.filename) != "" and not os.path.exists(
            os.path.dirname(self.args.filename)
        ):
            os.makedirs(os.path.dirname(self.args.filename))
        torch.save(self.model.state_dict(), self.args.filename)

    def load(self):
        """
        Load model under args.filename path.
        """
        if not os.path.exists(self.args.filename):
            raise Exception("Model file not found.")
        try:
            self.model.load_state_dict(torch.load(self.args.filename))
            self.model.to(self.device)
        except:
            raise Exception("Model file can't be loaded.")
