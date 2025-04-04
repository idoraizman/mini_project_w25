import abc

import torch.nn as nn
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

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
import torch.nn.functional as F


NUM_CLASSES = 10

def freeze_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
def get_args():   
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='Seed for random number generators')
    parser.add_argument('--data-path', default="/datasets/cv_datasets/data", type=str, help='Path to dataset')
    parser.add_argument('--batch-size', default=128, type=int, help='Size of each batch')
    parser.add_argument('--latent-dim', default=128, type=int, help='encoding dimension')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='Default device to use')
    parser.add_argument('--mnist', action='store_true', default=False,
                        help='Whether to use MNIST (True) or CIFAR10 (False) data')
    parser.add_argument('--self-supervised', action='store_true', default=False,
                        help='Whether train self-supervised with reconstruction objective, or jointly with classifier for classification objective.')
    parser.add_argument('--simclr', action='store_true', default=False)
    parser.add_argument('--val', action='store_true', default=False)
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs to train')
    parser.add_argument('--lr-ae', default=0.0002, type=float, help='Learning rate for autoencoder')
    parser.add_argument('--lr-cl', default=0.002, type=float, help='Learning rate for classifier')
    parser.add_argument('--dropout', default=0.2, type=float, help='Dropout rate')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature for NT-Xent loss')
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

    def __init__(self, model, loss_fn, optimizer, device):

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
            dl_val: DataLoader=None,
    ):

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
                self.model.load_state_dict(saved_state["model_state"])
                return None, best_acc

        for epoch in range(num_epochs):
            save_checkpoint = False
            verbose = False  # pass this to train/test_epoch.
            if epoch % print_every == 0 or epoch == num_epochs - 1:
                verbose = True
            self._print(f"--- EPOCH {epoch + 1}/{num_epochs} ---", verbose)

            train_result = self.train_epoch(dl_train, verbose=verbose)
            test_result = self.test_epoch(dl_test if dl_val is None else dl_val, verbose=verbose)
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

            if save_checkpoint and checkpoint_filename is not None:
                saved_state = dict(
                    best_acc=best_acc,
                    model_state=self.model.state_dict(),
                )
                torch.save(saved_state, checkpoint_filename)
                print(
                    f"*** Saved checkpoint {checkpoint_filename} " f"at epoch {epoch + 1}"
                )

            if post_epoch_fn:
                post_epoch_fn(epoch, train_result, test_result, verbose)

        return FitResult(actual_num_epochs, train_loss, train_acc, test_loss, test_acc), best_acc

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:

        self.model.train(True)  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:

        self.model.train(False)  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:

        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:

        raise NotImplementedError()

    @staticmethod
    def _print(message, verbose=True):
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(
            dl: DataLoader,
            forward_fn: Callable[[Any], BatchResult],
            verbose=True,
            max_batches=None,
    ) -> EpochResult:

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
        modules.append(nn.ReLU())
        modules.append(nn.Linear(in_features=128, out_features=10, bias=True))

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

    def __init__(self, model, loss_fn, optimizer, device, is_simclr=False):
        super().__init__(model, loss_fn, optimizer, device)
        self.is_simclr = is_simclr
    def train_batch(self, batch) -> BatchResult:
        x, y = batch
        if self.is_simclr:
            x = x[2]
        x = x.to(self.device)  # Image batch (N,C,H,W)
        y = y.to(self.device)  # Label batch (N,)

        y_pred = self.model(x)

        loss = self.loss_fn(y_pred, y)

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        predictions = torch.argmax(y_pred, dim=1)

        num_correct = (predictions == y).sum().item()

        return BatchResult(loss.item(), num_correct)

    def test_batch(self, batch) -> BatchResult:
        x, y = batch
        if self.is_simclr:
            x = x[2]
        x = x.to(self.device)  # Image batch (N,C,H,W)
        y = y.to(self.device)  # Label batch (N,)

        with torch.no_grad():

            y_pred = self.model(x)
            loss = self.loss_fn(y_pred, y)

            predictions = torch.argmax(y_pred, dim=1)
            num_correct = (predictions == y).sum().item()

        return BatchResult(loss.item(), num_correct)


class AETrainer(Trainer):
    def train_batch(self, batch) -> BatchResult:
        x, y = batch
        x = x.to(self.device)  # Image batch (N,C,H,W)
        y = y.to(self.device)  # Label batch (N,)

        output = self.model(x)

        loss = self.loss_fn(x, output)

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        return BatchResult(loss.item(), 1 / loss.item())

    def test_batch(self, batch) -> BatchResult:
        x, _ = batch
        x = x.to(self.device)  # Image batch (N,C,H,W)

        with torch.no_grad():

            output = self.model(x)
            loss = self.loss_fn(x, output)

        return BatchResult(loss.item(), 1 / loss.item())



class MnistEncoderCNN(nn.Module):
    def __init__(self, device, dropout=0.2):
        super().__init__()
        self.device = device
        modules = []

        modules.append(nn.Conv2d(1, 32, kernel_size=3))
        modules.append(nn.BatchNorm2d(32))
        modules.append(nn.PReLU())
        modules.append(nn.Dropout(dropout))

        modules.append(nn.Conv2d(32, 32, kernel_size=3, stride=2))
        modules.append(nn.BatchNorm2d(32))
        modules.append(nn.PReLU())
        modules.append(nn.Dropout(dropout))

        modules.append(nn.Conv2d(32, 64, kernel_size=3))
        modules.append(nn.BatchNorm2d(64))
        modules.append(nn.PReLU())
        modules.append(nn.Dropout(dropout))

        modules.append(nn.Flatten())
        modules.append(nn.Linear(in_features=6400, out_features=128, bias=True, device=self.device))

        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn(x)


class MnistDecoderCNN(nn.Module):
    def __init__(self, device, dropout=0.2):
        super().__init__()
        self.device = device
        modules = []

        modules.append(nn.Linear(in_features=128, out_features=6400, bias=True, device=self.device))
        modules.append(nn.BatchNorm1d(6400))
        modules.append(nn.PReLU())
        modules.append(nn.Dropout(dropout))
        modules.append(nn.Unflatten(1, (64, 10, 10)))

        modules.append(nn.ConvTranspose2d(64, 32, kernel_size=3))
        modules.append(nn.BatchNorm2d(32))
        modules.append(nn.ReLU())
        modules.append(nn.Dropout(dropout))

        modules.append(nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, output_padding=1))
        modules.append(nn.BatchNorm2d(32))
        modules.append(nn.ReLU())
        modules.append(nn.Dropout(dropout))

        modules.append(nn.ConvTranspose2d(32, 1, kernel_size=3))

        self.cnn = nn.Sequential(*modules)

    def forward(self, h):
        return torch.tanh(self.cnn(h))

class AE(nn.Module):
    def __init__(self, features_encoder, features_decoder):

        super().__init__()
        self.features_encoder = features_encoder
        self.features_decoder = features_decoder

    def encode(self, x):

        h = self.features_encoder(x)

        return h

    def decode(self, z):

        h = self.features_decoder(z)

        return h

    def forward(self, x):
        h = self.encode(x)
        return self.decode(h)


class CifarEncoderCNN(nn.Module):
    def __init__(self, device, dropout=0.2):
        super().__init__()
        self.device = device
        modules = []

        modules.append(nn.Conv2d(3, 32, kernel_size=3, padding=1))
        modules.append(nn.BatchNorm2d(32))
        modules.append(nn.PReLU())
        modules.append(nn.Dropout(dropout))

        modules.append(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1))
        modules.append(nn.BatchNorm2d(64))
        modules.append(nn.PReLU())
        modules.append(nn.Dropout(dropout))

        modules.append(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1))
        modules.append(nn.BatchNorm2d(128))
        modules.append(nn.PReLU())
        modules.append(nn.Dropout(dropout))

        modules.append(nn.Conv2d(128, 256, kernel_size=3, padding=1))
        modules.append(nn.BatchNorm2d(256))
        modules.append(nn.PReLU())
        modules.append(nn.Dropout(dropout))


        modules.append(nn.Flatten())
        modules.append(nn.Linear(in_features=16384, out_features=128, bias=True, device=self.device))

        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn(x)

class CifarDecoderCNN(nn.Module):
    def __init__(self, device, dropout=0.2):
        super().__init__()
        self.device = device
        modules = []

        modules.append(nn.Linear(in_features=128, out_features=16384, bias=True, device=self.device))
        modules.append(nn.BatchNorm1d(16384))
        modules.append(nn.PReLU())
        modules.append(nn.Dropout(dropout))
        modules.append(nn.Unflatten(1, (256, 8, 8)))

        modules.append(nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1))
        modules.append(nn.BatchNorm2d(128))
        modules.append(nn.ReLU())
        modules.append(nn.Dropout(dropout))

        modules.append(nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, output_padding=1, padding=1))
        modules.append(nn.BatchNorm2d(64))
        modules.append(nn.ReLU())
        modules.append(nn.Dropout(dropout))

        modules.append(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, output_padding=1, padding=1))
        modules.append(nn.BatchNorm2d(32))
        modules.append(nn.ReLU())
        modules.append(nn.Dropout(dropout))

        modules.append(nn.ConvTranspose2d(32, 3, kernel_size=3, padding=1))

        self.cnn = nn.Sequential(*modules)

    def forward(self, h):
        return torch.tanh(self.cnn(h))


def self_supervised_training(args, train_dl, test_dl, val_dl=None, test_dataset=None):
    encoder_model = MnistEncoderCNN(device=args.device, dropout=args.dropout).to(args.device) if args.mnist else CifarEncoderCNN(
        device=args.device).to(args.device)
    decoder_model = MnistDecoderCNN(device=args.device, dropout=args.dropout).to(args.device) if args.mnist else CifarDecoderCNN(
        device=args.device).to(args.device)

    ae = AE(encoder_model, decoder_model).to(args.device)

    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(ae.parameters(), lr=args.lr_ae, betas=(0.9, 0.999))

    trainer = AETrainer(model=ae, loss_fn=loss_fn, optimizer=optimizer, device=args.device)

    checkpoint_file_ae = 'mnist_ae' if args.mnist else 'cifar_ae'
    # checkpoint_file = None if args.val else checkpoint_file

    res, _ = trainer.fit(dl_train=train_dl, dl_test=test_dl, dl_val=val_dl, num_epochs=args.epochs, early_stopping=10, print_every=1,
                          checkpoints=checkpoint_file_ae)

    # Visualization section
    num_samples = 5
    random_indices = np.random.choice(len(test_dataset), num_samples)
    samples = [test_dataset[i][0].clone() for i in random_indices]
    samples = torch.stack(samples)
    samples = samples.to(args.device)
    reconstructions = ae(samples)

    samples_gpu = samples.clone() # Save GPU versions for interpolation

    samples = samples.detach().cpu()   # Move to CPU for visualization
    reconstructions = reconstructions.detach().cpu()

    fig, axes = plt.subplots(2, num_samples, figsize=(20, 4))
    for i in range(num_samples):
        sample = samples[i][0] if args.mnist else samples[i].permute(1, 2, 0)
        sample_reco = reconstructions[i][0] if args.mnist else reconstructions[i].permute(1, 2, 0)
        axes[0, i].imshow(sample, cmap='gray' if args.mnist else None)
        axes[1, i].imshow(sample_reco, cmap='gray' if args.mnist else None)
    plt.savefig('mnist_reconstructions.png' if args.mnist else 'cifar_reconstructions.png')
    plt.show()

    # Interpolation
    def interpolate(a, b, steps):
        return torch.stack([a + (b - a) * (i / (steps - 1)) for i in range(steps)])

    # Use the GPU version of samples
    a = encoder_model(samples_gpu[0].unsqueeze(0))
    b = encoder_model(samples_gpu[1].unsqueeze(0))
    inter = interpolate(a, b, 10).squeeze(1)
    reconstructions = decoder_model(inter)
    reconstructions = reconstructions.detach().cpu()

    fig, axes = plt.subplots(1, 10, figsize=(20, 4))
    for i in range(10):
        img = reconstructions[i][0] if args.mnist else reconstructions[i].permute(1, 2, 0)
        axes[i].imshow(img, cmap='gray' if args.mnist else None)
    plt.savefig('mnist_interpolation.png' if args.mnist else 'cifar_interpolation.png')

    classifier = Classifier(encoder_model).to(args.device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.classifier.parameters(), lr=args.lr_cl, betas=(0.9, 0.999))
    classifier_trainer = ClassifierTrainer(model=classifier, loss_fn=loss_fn, optimizer=optimizer,
                                           device=args.device, is_simclr=False)

    checkpoint_file = "mnist_classifier" if args.mnist else "cifar_classifier"
    checkpoint_file = None if args.val else checkpoint_file

    res, res_best_acc = classifier_trainer.fit(dl_train=train_dl, dl_test=test_dl, dl_val=val_dl, num_epochs=args.epochs, early_stopping=10,
                                    print_every=1, checkpoints=checkpoint_file)

    plot_tsne(encoder_model, test_dl, 'self_supervised_' + ("mnist" if args.mnist else "cifar"), False, args.device)

    return res_best_acc, checkpoint_file_ae

def supervised_training(args, train_dl, test_dl, val_dl=None):
    encoder_model = MnistEncoderCNN(device=args.device, dropout=args.dropout).to(args.device) if args.mnist else CifarEncoderCNN(
        device=args.device).to(args.device)
    classifier = Classifier(encoder_model, freeze_encoder=False).to(args.device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr_cl, betas=(0.9, 0.999))

    trainer = ClassifierTrainer(model=classifier, loss_fn=loss_fn, optimizer=optimizer, device=args.device)

    checkpoint_file = 'mnist_classifier_supervised' if args.mnist else 'cifar_classifier_supervised'
    checkpoint_file = None if args.val else checkpoint_file
    res, res_best_acc = trainer.fit(dl_train=train_dl, dl_test=test_dl, dl_val=val_dl, num_epochs=args.epochs, early_stopping=10, print_every=1,
                          checkpoints=checkpoint_file)

    plot_tsne(encoder_model, test_dl, 'supervised_' + ("mnist" if args.mnist else "cifar"), False, args.device)

    return res_best_acc, None


class SimCLRTransform:

    def __init__(self, size=32):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.regular_transform = transforms.Compose([transforms.ToTensor()])

    def __call__(self, x):
        return [self.transform(x), self.transform(x), self.regular_transform(x)]


class SimCLR(nn.Module):
    def __init__(self, device, hidden_dim=128, is_mnist=False):

        super(SimCLR, self).__init__()
        self.device = device
        self.encoder = torchvision.models.resnet50(pretrained=False)

        if is_mnist:
            self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.encoder.fc = nn.Identity()

        self.projection = nn.Sequential(
            nn.Linear(2048, 4 * hidden_dim, device=self.device),
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim, device=self.device),
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection(h)
        return z

def nt_xent_loss(z_i, z_j, temperature=0.5):

    batch_size = z_i.size(0)

    z = torch.cat([z_i, z_j], dim=0)
    z = F.normalize(z, dim=1)

    # Compute similarity matrix (cosine similarity)
    similarity_matrix = torch.matmul(z, z.T)

    mask = torch.eye(2 * batch_size, device=z.device).bool()  # filter out self similarities
    similarity_matrix.masked_fill_(mask, -torch.inf)

    positives = torch.cat([
        torch.diag(similarity_matrix, batch_size),
        torch.diag(similarity_matrix, -batch_size)
    ])

    numerator = torch.exp(positives / temperature)
    denominator = torch.sum(torch.exp(similarity_matrix / temperature), dim=1)
    loss = -torch.log(numerator / denominator)
    return loss.mean()

def get_nt_xent_loss(temperature=0.5):
    return lambda z_i, z_j: nt_xent_loss(z_i, z_j, temperature=temperature)

class SimCLRTrainer(Trainer):
    def train_batch(self, batch) -> BatchResult:
        x, y = batch
        x_i, x_j, _ = x
        x_i = x_i.to(self.device)
        x_j = x_j.to(self.device)

        z_i = self.model(x_i)
        z_j = self.model(x_j)

        loss = self.loss_fn(z_i, z_j)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return BatchResult(loss.item(), 1/loss.item())

    def test_batch(self, batch) -> BatchResult:
        x, y = batch
        x_i, x_j, _ = x
        x_i = x_i.to(self.device)
        x_j = x_j.to(self.device)

        z_i = self.model(x_i)
        z_j = self.model(x_j)

        loss = self.loss_fn(z_i, z_j)

        return BatchResult(loss.item(), 1/loss.item())

def simclr_training(args, train_dl, test_dl, val_dl=None):
    simclr = SimCLR(hidden_dim=args.latent_dim, is_mnist=args.mnist, device=args.device)
    loss_fn = get_nt_xent_loss(args.temperature)
    optimizer = torch.optim.Adam(simclr.parameters(), lr=args.lr_ae, betas=(0.9, 0.999))
    trainer = SimCLRTrainer(model=simclr, loss_fn=loss_fn, optimizer=optimizer, device=args.device)
    checkpoint_file_simclr = "simclr_mnist" if args.mnist else "simclr_cifar"
    # checkpoint_file = None if args.val else checkpoint_file
    res, _ = trainer.fit(dl_train=train_dl, dl_test=test_dl, dl_val=val_dl, num_epochs=args.epochs, early_stopping=10, print_every=1, checkpoints=checkpoint_file_simclr)

    classifier = Classifier(simclr, freeze_encoder=True).to(args.device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.classifier.parameters(), lr=args.lr_cl, betas=(0.9, 0.999))
    classifier_trainer = ClassifierTrainer(model=classifier, loss_fn=loss_fn, optimizer=optimizer,
                                           device=args.device, is_simclr=True)

    classifier_checkpoint_file = "simclr_mnist_classifier" if args.mnist else "simclr_cifar_classifier"
    classifier_checkpoint_file = None if args.val else classifier_checkpoint_file

    res, res_best_acc = classifier_trainer.fit(dl_train=train_dl, dl_test=test_dl, dl_val=val_dl, num_epochs=args.epochs, early_stopping=10,
                                 print_every=1, checkpoints=classifier_checkpoint_file)

    if not args.val:
        plot_tsne(simclr, test_dl, 'simclr_' + ("mnist" if args.mnist else "cifar"), True, args.device)

    return res_best_acc, checkpoint_file_simclr

def tune_hp(args, transform):
    best_acc = 0
    best_hp = {}
    temperatures = [0.1] if args.simclr else [0.1]
    for temperature in temperatures: # update hyper parameters values when necessary
        for lr_ae in [0.00008]:
            checkpoint_ae = None
            for lr_cl in [0.0002]:
                for dropout in [0.2]:
                    for batch_size in [64]:

                        args.lr_ae = lr_ae
                        args.lr_cl = lr_cl
                        args.dropout = dropout
                        args.batch_size = batch_size
                        args.temperature = temperature
                        print("Hyperparameters:", {"lr_ae": lr_ae, "lr_cl": lr_cl, "dropout": dropout, "batch_size": batch_size, "temperature": temperature}, flush=True)

                        if args.mnist:
                            train_dataset = datasets.MNIST(root=args.data_path, train=True, download=False,
                                                           transform=transform)
                            test_dataset = datasets.MNIST(root=args.data_path, train=False, download=False,
                                                          transform=transform)
                        else:
                            train_dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True,
                                                             transform=transform)
                            test_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True,
                                                            transform=transform)

                        val_ratio = 0.1
                        train_size = int((1 - val_ratio) * len(train_dataset))  # 90% train
                        val_size = len(train_dataset) - train_size  # 10% validation

                        # Perform the split
                        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
                        train_dl = torch.utils.data.DataLoader(
                            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
                        )
                        val_dl = torch.utils.data.DataLoader(
                            val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
                        )
                        test_dl = torch.utils.data.DataLoader(
                            test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
                        )

                        if args.simclr:
                            cur_best_acc, checkpoint_ae = simclr_training(args, train_dl, test_dl, val_dl)
                        elif args.self_supervised:
                            cur_best_acc, checkpoint_ae = self_supervised_training(args, train_dl, test_dl, val_dl, test_dataset)
                        else:
                            cur_best_acc, checkpoint_ae = supervised_training(args, train_dl, test_dl, val_dl)

                        if cur_best_acc > best_acc:
                            best_acc = cur_best_acc
                            best_hp = {"lr_ae": lr_ae, "lr_cl": lr_cl, "dropout": dropout, "batch_size": batch_size, "temperature": temperature}
                        print("Current hyper parameters best accuracy: ", cur_best_acc, flush=True)
            if checkpoint_ae is not None:
                os.remove(checkpoint_ae + '.pt')
    print("Best hyperparameters:", best_hp, flush=True)
    print("Best accuracy:", best_acc, flush=True)
    return best_hp


if __name__ == "__main__":


    args = get_args()
    freeze_seeds(args.seed)

    if args.simclr:
        size = 28 if args.mnist else 32
        transform = SimCLRTransform(size=size)
    elif args.mnist:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),

        ])

    print("Device:", args.device)
    if args.val:
        tune_hp(args, transform)
        exit()

    if args.mnist:
        train_dataset = datasets.MNIST(root=args.data_path, train=True, download=False,
                                       transform=transform)
        test_dataset = datasets.MNIST(root=args.data_path, train=False, download=False,
                                      transform=transform)
    else:
        train_dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True,
                                         transform=transform)
        test_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True,
                                        transform=transform)

    # Perform the split
    train_dl = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    test_dl = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )

    if args.simclr:
        res, _ = simclr_training(args, train_dl, test_dl)
    elif args.self_supervised:
        res, _ = self_supervised_training(args, train_dl, test_dl, test_dataset=test_dataset)
    else:
        res, _ = supervised_training(args, train_dl, test_dl)

    print("Best accuracy:", res, flush=True)














