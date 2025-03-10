import abc

import torch.nn as nn
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

import numpy as np
from matplotlib import pyplot as plt
from utils import plot_tsne
import numpy as np
import random
import argparse
import os
from pathlib import Path
from typing import List, NamedTuple, Callable, Any
import tqdm
import sys


NUM_CLASSES = 10

def freeze_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
def get_args():   
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='Seed for random number generators')
    parser.add_argument('--data-path', default="/datasets/cv_datasets/data", type=str, help='Path to dataset')
    parser.add_argument('--batch-size', default=64, type=int, help='Size of each batch')
    parser.add_argument('--latent-dim', default=128, type=int, help='encoding dimension')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='Default device to use')
    parser.add_argument('--mnist', action='store_true', default=True,
                        help='Whether to use MNIST (True) or CIFAR10 (False) data')
    parser.add_argument('--self-supervised', action='store_true', default=False,
                        help='Whether train self-supervised with reconstruction objective, or jointly with classifier for classification objective.')
    return parser.parse_args()

class BatchResult(NamedTuple):
    """
    Represents the result of training for a single batch: the loss
    and number of correct classifications.
    """

    loss: float
    num_correct: int


class EpochResult(NamedTuple):
    """
    Represents the result of training for a single epoch: the loss per batch
    and accuracy on the dataset (train or test).
    """

    losses: List[float]
    accuracy: float


class FitResult(NamedTuple):
    """
    Represents the result of fitting a model for multiple epochs given a
    training and test (or validation) set.
    The losses are for each batch and the accuracies are per epoch.
    """

    num_epochs: int
    train_loss: List[float]
    train_acc: List[float]
    test_loss: List[float]
    test_acc: List[float]

class Trainer(abc.ABC):
    """
    A class abstracting the various tasks of training models.

    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(self, model, loss_fn, optimizer, device):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        model.to(self.device)

    def fit(
            self,
            dl_train: DataLoader,
            dl_test: DataLoader,
            num_epochs,
            checkpoints: str = None,
            early_stopping: int = None,
            print_every=1,
            post_epoch_fn=None,
            **kw,
    ) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :param post_epoch_fn: A function to call after each epoch completes.
        :return: A FitResult object containing train and test losses per epoch.
        """
        actual_num_epochs = 0
        train_loss, train_acc, test_loss, test_acc = [], [], [], []

        best_acc = None
        epochs_without_improvement = 0

        checkpoint_filename = None
        if checkpoints is not None:
            checkpoint_filename = f"{checkpoints}.pt"
            Path(os.path.dirname(checkpoint_filename)).mkdir(exist_ok=True)
            if os.path.isfile(checkpoint_filename):
                print(f"*** Loading checkpoint file {checkpoint_filename}")
                saved_state = torch.load(checkpoint_filename, map_location=self.device)
                best_acc = saved_state.get("best_acc", best_acc)
                epochs_without_improvement = saved_state.get(
                    "ewi", epochs_without_improvement
                )
                self.model.load_state_dict(saved_state["model_state"])

        for epoch in range(num_epochs):
            save_checkpoint = False
            verbose = False  # pass this to train/test_epoch.
            if epoch % print_every == 0 or epoch == num_epochs - 1:
                verbose = True
            self._print(f"--- EPOCH {epoch + 1}/{num_epochs} ---", verbose)

            # TODO:
            #  Train & evaluate for one epoch
            #  - Use the train/test_epoch methods.
            #  - Save losses and accuracies in the lists above.
            #  - Implement early stopping. This is a very useful and
            #    simple regularization technique that is highly recommended.
            # ====== YOUR CODE: ======
            train_result = self.train_epoch(dl_train, verbose=verbose)
            test_result = self.test_epoch(dl_test, verbose=verbose)
            # print(train_res, test_res)
            train_loss.append(sum(train_result.losses) / len(train_result.losses))
            train_acc.append(train_result.accuracy)
            test_loss.append(sum(test_result.losses) / len(test_result.losses))
            test_acc.append(test_result.accuracy)
            if best_acc is None or test_result.accuracy > best_acc:
                best_acc = test_result.accuracy
                if checkpoints:
                    save_checkpoint = True
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if early_stopping is not None and epochs_without_improvement == early_stopping:
                    break
            # ========================

            # Save model checkpoint if requested
            if save_checkpoint and checkpoint_filename is not None:
                saved_state = dict(
                    best_acc=best_acc,
                    ewi=epochs_without_improvement,
                    model_state=self.model.state_dict(),
                )
                torch.save(saved_state, checkpoint_filename)
                print(
                    f"*** Saved checkpoint {checkpoint_filename} " f"at epoch {epoch + 1}"
                )

            if post_epoch_fn:
                post_epoch_fn(epoch, train_result, test_result, verbose)

        return FitResult(actual_num_epochs, train_loss, train_acc, test_loss, test_acc)

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(
            dl: DataLoader,
            forward_fn: Callable[[Any], BatchResult],
            verbose=True,
            max_batches=None,
    ) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        num_correct = 0
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_file = sys.stdout
        else:
            pbar_file = open(os.devnull, "w")

        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches, file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data)

                pbar.set_description(f"{pbar_name} ({batch_res.loss:.3f})")
                pbar.update()

                losses.append(batch_res.loss)
                num_correct += batch_res.num_correct

            avg_loss = sum(losses) / num_batches
            accuracy = 100.0 * num_correct / num_samples
            pbar.set_description(
                f"{pbar_name} "
                f"(Avg. Loss {avg_loss:.3f}, "
                f"Accuracy {accuracy:.1f})"
            )

        return EpochResult(losses=losses, accuracy=accuracy)


class Classifier(nn.Module):
    def __init__(self, encoder_model, freeze_encoder=True):
        super().__init__()
        self.freeze_encoder = freeze_encoder
        self.encoder_model = encoder_model
        if freeze_encoder:
            for param in self.encoder_model.parameters():
                param.requires_grad = False

        modules = []
        modules.append(nn.Linear(in_features=128, out_features=128, bias=True))
        # modules.append(nn.BatchNorm1d(128))
        modules.append(nn.ReLU())
        modules.append(nn.Dropout(0.2))
        modules.append(nn.Linear(in_features=128, out_features=128, bias=True))

        self.classifier = nn.Sequential(*modules)

    def forward(self, x):
        if self.freeze_encoder:
            with torch.no_grad():
                x = self.encoder_model(x)
        else:
            x = self.encoder_model(x)
        return self.classifier(x)

    def predict(self, x):
        return torch.argmax(self.forward(x), dim=1)


class ClassifierTrainer(Trainer):
    def train_batch(self, batch) -> BatchResult:
        x, y = batch
        x = x.to(self.device)  # Image batch (N,C,H,W)
        y = y.to(self.device)  # Label batch (N,)

        # ====== YOUR CODE: ======
        y_pred = self.model(x)

        loss = self.loss_fn(y_pred, y)

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        # ========================
        predictions = torch.argmax(y_pred, dim=-1)
        accuracy = (predictions == y).sum().item()

        return BatchResult(loss.item(), accuracy / 64)

    def test_batch(self, batch) -> BatchResult:
        x, y = batch
        x = x.to(self.device)  # Image batch (N,C,H,W)
        y = y.to(self.device)  # Label batch (N,)

        with torch.no_grad():

            # ====== YOUR CODE: ======
            y_pred = self.model(x)

            loss = self.loss_fn(y_pred, y)
            # ========================
            predictions = torch.argmax(y_pred, dim=-1)
            accuracy = (predictions == y).sum().item()

        return BatchResult(loss.item(), accuracy/y.shape[0])


class AETrainer(Trainer):
    def train_batch(self, batch) -> BatchResult:
        x, _y = batch
        x = x.to(self.device)  # Image batch (N,C,H,W)

        # ====== YOUR CODE: ======
        output = self.model(x)

        loss = self.loss_fn(x, output)

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        # ========================

        return BatchResult(loss.item(), 1 / loss.item())

    def test_batch(self, batch) -> BatchResult:
        x, _ = batch
        x = x.to(self.device)  # Image batch (N,C,H,W)

        with torch.no_grad():

            # ====== YOUR CODE: ======
            output = self.model(x)

            loss = self.loss_fn(x, output)
            # ========================

        return BatchResult(loss.item(), 1 / loss.item())



class MnistEncoderCNN(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        modules = []

        # ====== YOUR CODE: ======
        modules.append(nn.Conv2d(1, 32, kernel_size=3))
        modules.append(nn.BatchNorm2d(32))
        modules.append(nn.PReLU())
        modules.append(nn.Dropout(0.4))

        modules.append(nn.Conv2d(32, 32, kernel_size=3, stride=2))
        modules.append(nn.BatchNorm2d(32))
        modules.append(nn.PReLU())
        modules.append(nn.Dropout(0.4))

        modules.append(nn.Conv2d(32, 64, kernel_size=3))
        modules.append(nn.BatchNorm2d(64))
        modules.append(nn.PReLU())
        modules.append(nn.Dropout(0.4))

        modules.append(nn.Flatten())
        modules.append(nn.Linear(in_features=6400, out_features=128, bias=True, device=self.device))

        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn(x)


class MnistDecoderCNN(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        modules = []


        # ====== YOUR CODE: ======
        modules.append(nn.Linear(in_features=128, out_features=6400, bias=True, device=self.device))
        modules.append(nn.BatchNorm1d(6400))
        modules.append(nn.PReLU())
        modules.append(nn.Dropout(0.4))
        modules.append(nn.Unflatten(1, (64, 10, 10)))

        modules.append(nn.ConvTranspose2d(64, 32, kernel_size=3))
        modules.append(nn.BatchNorm2d(32))
        modules.append(nn.ReLU())
        modules.append(nn.Dropout(0.4))

        modules.append(nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, output_padding=1))
        modules.append(nn.BatchNorm2d(32))
        modules.append(nn.ReLU())
        modules.append(nn.Dropout(0.4))

        modules.append(nn.ConvTranspose2d(32, 1, kernel_size=3))
        # ========================

        self.cnn = nn.Sequential(*modules)

    def forward(self, h):
        # Tanh to scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(self.cnn(h))

class AE(nn.Module):
    def __init__(self, features_encoder, features_decoder):
        """
        :param features_encoder: Instance of an encoder the extracts features
        from an input.
        :param features_decoder: Instance of a decoder that reconstructs an
        input from it's features.
        """
        super().__init__()
        self.features_encoder = features_encoder
        self.features_decoder = features_decoder

    def encode(self, x):

        # ====== YOUR CODE: ======
        h = self.features_encoder(x)
        # ========================

        return h

    def decode(self, z):

        # ====== YOUR CODE: ======
        h = self.features_decoder(z)
        # ========================

        return h

    def sample(self, n):  # TODO: change
        samples = []
        device = next(self.parameters()).device
        with torch.no_grad():

            # ====== YOUR CODE: ======
            for _ in range(n):
                sample = torch.randn(size=(1,self.z_dim), device=device)
                sample_recon = self.decode(sample)
                samples.append(sample_recon.squeeze(0))
            # ========================

        # Detach and move to CPU for display purposes
        samples = [s.detach().cpu() for s in samples]
        return samples

    def forward(self, x):
        h = self.encode(x)
        return self.decode(h)


class CifarEncoderCNN(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        modules = []

        # ====== YOUR CODE: ======
        modules.append(nn.Conv2d(3, 32, kernel_size=3, padding=1))
        modules.append(nn.BatchNorm2d(32))
        modules.append(nn.PReLU())
        modules.append(nn.Dropout(0.4))

        modules.append(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1))
        modules.append(nn.BatchNorm2d(64))
        modules.append(nn.PReLU())
        modules.append(nn.Dropout(0.4))

        modules.append(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1))
        modules.append(nn.BatchNorm2d(128))
        modules.append(nn.PReLU())
        modules.append(nn.Dropout(0.4))

        modules.append(nn.Conv2d(128, 256, kernel_size=3, padding=1))
        modules.append(nn.BatchNorm2d(256))
        modules.append(nn.PReLU())
        modules.append(nn.Dropout(0.4))


        modules.append(nn.Flatten())
        modules.append(nn.Linear(in_features=16384, out_features=128, bias=True, device=self.device))

        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn(x)

class CifarDecoderCNN(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        modules = []


        # ====== YOUR CODE: ======
        modules.append(nn.Linear(in_features=128, out_features=16384, bias=True, device=self.device))
        modules.append(nn.BatchNorm1d(16384))
        modules.append(nn.PReLU())
        modules.append(nn.Dropout(0.4))
        modules.append(nn.Unflatten(1, (256, 8, 8)))

        modules.append(nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1))
        modules.append(nn.BatchNorm2d(128))
        modules.append(nn.ReLU())
        modules.append(nn.Dropout(0.4))

        modules.append(nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, output_padding=1, padding=1))
        modules.append(nn.BatchNorm2d(64))
        modules.append(nn.ReLU())
        modules.append(nn.Dropout(0.4))

        modules.append(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, output_padding=1, padding=1))
        modules.append(nn.BatchNorm2d(32))
        modules.append(nn.ReLU())
        modules.append(nn.Dropout(0.4))

        modules.append(nn.ConvTranspose2d(32, 3, kernel_size=3))
        # ========================

        self.cnn = nn.Sequential(*modules)

    def forward(self, h):
        # Tanh to scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(self.cnn(h))


def self_supervised_training(args, train_dl, test_dl, val_dl, train_dataset, test_dataset):
    encoder_model = MnistEncoderCNN(device=args.device).to(args.device) if args.mnist else CifarEncoderCNN(
        device=args.device).to(args.device)
    decoder_model = MnistDecoderCNN(device=args.device).to(args.device) if args.mnist else CifarDecoderCNN(
        device=args.device).to(args.device)

    ae = AE(encoder_model, decoder_model)

    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(ae.parameters(), lr=10 ** -3, betas=(0.9, 0.999))

    trainer = AETrainer(model=ae, loss_fn=loss_fn, optimizer=optimizer, device=args.device)

    checkpoint_file = 'mnist_ae' if args.mnist else 'cifar_ae'

    if os.path.isfile(f'{checkpoint_file}.pt'):
        print(f'*** Loading final checkpoint file {checkpoint_file} instead of training')

    else:
        res = trainer.fit(dl_train=train_dl, dl_test=test_dl, num_epochs=100, early_stopping=10, print_every=1,
                          checkpoints='mnist_ae')

    # Plot images from best model
    saved_state = torch.load(f'{checkpoint_file}.pt', map_location=args.device)
    ae.load_state_dict(saved_state['model_state'])

    num_samples = 5
    random_indices = np.random.choice(len(test_dataset), num_samples)
    samples = [test_dataset[i][0] for i in random_indices]
    samples = torch.stack(samples)
    samples = samples.to(args.device)
    reconstructions = ae(samples)
    samples = samples.detach().cpu()
    reconstructions = reconstructions.detach().cpu()
    fig, axes = plt.subplots(2, num_samples, figsize=(20, 4))
    for i in range(num_samples):
        axes[0, i].imshow(samples[i][0], cmap='gray')
        axes[1, i].imshow(reconstructions[i][0], cmap='gray')
    # saving images
    plt.savefig('mnist_reconstructions.png' if args.mnist else 'cifar_reconstructions.png')
    plt.show()

    # interpolation
    def interpolate(a, b, steps):
        return torch.stack([a + (b - a) * (i / (steps - 1)) for i in range(steps)])

    a = encoder_model(samples[0].unsqueeze(0))
    b = encoder_model(samples[1].unsqueeze(0))
    inter = interpolate(a, b, 10).squeeze(1)
    reconstructions = decoder_model(inter)
    reconstructions = reconstructions.detach().cpu()
    fig, axes = plt.subplots(1, 10, figsize=(20, 4))
    for i in range(10):
        img = reconstructions[i][0] if args.mnist else reconstructions[i].permute(1, 2, 0)
        axes[i].imshow(img.cpu().detach().numpy(), cmap='gray' if args.mnist else None)
    plt.savefig('mnist_interpolation.png' if args.mnist else 'cifar_interpolation.png')

    classifier = Classifier(encoder_model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.classifier.parameters(), lr=10 ** -4, betas=(0.9, 0.999))
    classifier_trainer = ClassifierTrainer(model=classifier, loss_fn=loss_fn, optimizer=optimizer,
                                           device=args.device)

    checkpoint_file = "mnist_classifier" if args.mnist else "cifar_classifier"
    if os.path.isfile(f'{checkpoint_file}.pt'):
        print(f'*** Loading final checkpoint file {checkpoint_file} instead of training')
    else:
        res = classifier_trainer.fit(dl_train=train_dl, dl_test=test_dl, num_epochs=100, early_stopping=10,
                                     print_every=1, checkpoints=checkpoint_file)

def supervised_training(args, train_dl, test_dl, val_dl, train_dataset, test_dataset):
    encoder_model = MnistEncoderCNN(device=args.device).to(args.device) if args.mnist else CifarEncoderCNN(
        device=args.device).to(args.device)
    classifier = Classifier(encoder_model, freeze_encoder=False)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=10 ** -3, betas=(0.9, 0.999))

    trainer = ClassifierTrainer(model=classifier, loss_fn=loss_fn, optimizer=optimizer, device=args.device)

    checkpoint_file = 'mnist_classifier_supervised' if args.mnist else 'cifar_classifier_supervised'

    if os.path.isfile(f'{checkpoint_file}.pt'):
        print(f'*** Loading final checkpoint file {checkpoint_file} instead of training')

    else:
        res = trainer.fit(dl_train=train_dl, dl_test=test_dl, num_epochs=100, early_stopping=10, print_every=1,
                          checkpoints=checkpoint_file)



if __name__ == "__main__":


    args = get_args()
    freeze_seeds(args.seed)

    if args.mnist:
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 # std=[0.5, 0.5, 0.5])
            # one possible convenient normalization. You don't have to use it.

        ])


    print("Device:", args.device)

    # tuning:
    args.batch_size = 64
                                           
    if args.mnist:
        train_dataset = datasets.MNIST(root=args.data_path, train=True, download=False, transform=transform)
        test_dataset = datasets.MNIST(root=args.data_path, train=False, download=False, transform=transform)
    else:
        train_dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform)
        
    # When you create your dataloader you should split train_dataset or test_dataset to leave some aside for validation
    val_ratio = 0.1
    train_size = int((1 - val_ratio) * len(train_dataset))  # 90% train
    val_size = len(train_dataset) - train_size  # 10% validation

    # Perform the split
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    train_dl = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1
    )
    val_dl = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1
    )
    test_dl = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1
    )

    if args.self_supervised:
        self_supervised_training(args, train_dl, test_dl, val_dl, train_dataset, test_dataset)
    else:
        supervised_training(args, train_dl, test_dl, val_dl, train_dataset, test_dataset)










