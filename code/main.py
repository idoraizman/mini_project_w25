import torch.nn as nn
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split

import numpy as np
from matplotlib import pyplot as plt
from utils import plot_tsne
import numpy as np
import random
import argparse

NUM_CLASSES = 10

def freeze_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
def get_args():   
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='Seed for random number generators')
    parser.add_argument('--data-path', default="/datasets/cv_datasets/data", type=str, help='Path to dataset')
    parser.add_argument('--batch-size', default=8, type=int, help='Size of each batch')
    parser.add_argument('--latent-dim', default=128, type=int, help='encoding dimension')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='Default device to use')
    parser.add_argument('--mnist', action='store_true', default=True,
                        help='Whether to use MNIST (True) or CIFAR10 (False) data')
    parser.add_argument('--self-supervised', action='store_true', default=False,
                        help='Whether train self-supervised with reconstruction objective, or jointly with classifier for classification objective.')
    return parser.parse_args()



class MnistEncoderCNN(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        modules = []

        # TODO:
        #  Implement a CNN. Save the layers in the modules list.
        #  The input shape is an image batch: (N, in_channels, H_in, W_in).
        #  The output shape should be (N, out_channels, H_out, W_out).
        #  You can assume H_in, W_in >= 64.
        #  Architecture is up to you, but it's recommended to use at
        #  least 3 conv layers. You can use any Conv layer parameters,
        #  use pooling or only strides, use any activation functions,
        #  use BN or Dropout, etc.
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

        # TODO:
        #  Implement the "mirror" CNN of the encoder.
        #  For example, instead of Conv layers use transposed convolutions,
        #  instead of pooling do unpooling (if relevant) and so on.
        #  The architecture does not have to exactly mirror the encoder
        #  (although you can), however the important thing is that the
        #  output should be a batch of images, with same dimensions as the
        #  inputs to the Encoder were.
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


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  #one possible convenient normalization. You don't have to use it.

    ])

    args = get_args()
    freeze_seeds(args.seed)
                
                                           
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
    #this is just for the example. Simple flattening of the image is probably not the best idea                                        
    encoder_model = MnistEncoderCNN(device=args.device).to(args.device)
    decoder_model = MnistDecoderCNN(device=args.device).to(args.device)

    sample = torch.stack([train_dataset[i][0] for i in range(2)]).to(args.device) #This is just for the example - you should use a dataloader
    output = decoder_model(encoder_model(sample))
    print("Sample shape: ", sample.shape)
    print(output.shape)

