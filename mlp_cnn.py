import random
import os
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor, normalize, affine
from PIL import Image
from typing import Tuple, List, NamedTuple
from tqdm import tqdm
import zipfile
from urllib import request

# Seed all random number generators
np.random.seed(197331)
torch.manual_seed(197331)
random.seed(197331)


class NetworkConfiguration(NamedTuple):
    n_channels: Tuple[int, ...] = (16, 32, 48)
    kernel_sizes: Tuple[int, ...] = (3, 3, 3)
    strides: Tuple[int, ...] = (1, 1, 1)
    dense_hiddens: Tuple[int, ...] = (256, 256)


class Trainer:
    def __init__(self,
                 network_type: str = "mlp",
                 net_config: NetworkConfiguration = NetworkConfiguration(),
                 lr: float = 0.001,
                 batch_size: int = 128,
                 activation_name: str = "relu"):
        self.train, self.test = self.load_dataset()
        self.network_type = network_type
        activation_function = self.create_activation_function(activation_name)
        input_dim = self.train[0].shape[1:]
        if network_type == "mlp":
            self.network = self.create_mlp(input_dim[0] * input_dim[1] * input_dim[2],
                                           net_config,
                                           activation_function)
        elif network_type == "cnn":
            self.network = self.create_cnn(input_dim[0],
                                           net_config,
                                           activation_function)
        else:
            raise ValueError("Network type not supported")
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.lr = lr
        self.batch_size = batch_size

        self.train_logs = {'train_loss': [], 'test_loss': [],
                           'train_mae': [], 'test_mae': []}

    @staticmethod
    def load_dataset() -> Tuple[
        Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        if not os.path.exists('./rotated_fashion_mnist'):
            url = 'https://drive.google.com/u/0/uc?id=1NQPmr01eIafQKeH9C9HR0lGuB5z6mhGb&export=download&confirm=t&uuid=645ff20a-d47b-49f0-ac8b-4a7347529c8e&at=AHV7M3d_Da0D7wowJlTzzZxDky5c:1669325231545'
            with request.urlopen(url) as f:
                with open('./rotated_fashion_mnist.zip', 'wb') as out:
                    out.write(f.read())
            with zipfile.ZipFile('./rotated_fashion_mnist.zip', 'r') as zip_ref:
                zip_ref.extractall()
            os.remove('./rotated_fashion_mnist.zip')

        datapath = './rotated_fashion_mnist'

        def get_paths_and_rots(split: str) -> List[Tuple[str, float]]:
            image_paths, rots = [], []
            files = os.listdir(os.path.join(datapath, split))
            for file in files:
                image_paths.append(os.path.join(datapath, split, file))
                rots.append(float(file.split('_')[1].split('.')[0]))
            return image_paths, rots

        def to_tensors(image_paths: List[str], rots: List[float]) -> Tuple[torch.Tensor, torch.Tensor]:
            images = [normalize(to_tensor(Image.open(path)), (0.5,), (0.5,))
                      for path in image_paths]
            images = torch.stack(images)
            labels = torch.tensor(rots).view(-1, 1)
            return images, labels

        X_train, y_train = to_tensors(*get_paths_and_rots('train'))
        X_test, y_test = to_tensors(*get_paths_and_rots('test'))

        # Normalize y for easier training
        mean, std = y_train.mean(), y_train.std()
        y_train = (y_train - mean) / std
        y_test = (y_test - mean) / std

        return (X_train, y_train), (X_test, y_test)

    @staticmethod
    def create_mlp(input_dim: int, net_config: NetworkConfiguration,
                   activation: torch.nn.Module) -> torch.nn.Module:
        """
        Create a multi-layer perceptron (MLP) network.

        :param net_config: a NetworkConfiguration named tuple. Only the field 'dense_hiddens' will be used.
        :param activation: The activation function to use.
        :return: A PyTorch model implementing the MLP.
        """
        modules = []
        nb_neurons = [input_dim] + [n for n in net_config.dense_hiddens]
        modules.append(torch.nn.Flatten())

        #Linear layers and activation
        for i in range(len(nb_neurons)-1):
            modules.append(torch.nn.Linear(nb_neurons[i], nb_neurons[i+1]))
            modules.append(activation)

        #Output layer
        modules.append(torch.nn.Linear(nb_neurons[-1], 1))
        return torch.nn.Sequential(*modules)

    @staticmethod
    def create_cnn(in_channels: int, net_config: NetworkConfiguration,
                   activation: torch.nn.Module) -> torch.nn.Module:
        """
        Create a convolutional network.

        :param in_channels: The number of channels in the input image.
        :param net_config: a NetworkConfiguration specifying the architecture of the CNN.
        :param activation: The activation function to use.
        :return: A PyTorch model implementing the CNN.
        """
        channels = [in_channels] + [c for c in net_config.n_channels]
        modules = []
        #Couches convolutives
        for i in range(len(channels)-2):
            modules.append(torch.nn.Conv2d(channels[i], channels[i+1], net_config.kernel_sizes[i], net_config.strides[i]))
            modules.append(activation)
            modules.append(torch.nn.MaxPool2d(2))
        #Dernière couche convolutive:
        modules.append(torch.nn.Conv2d(net_config.n_channels[-2], net_config.n_channels[-1], net_config.kernel_sizes[-1], net_config.strides[-1]))
        modules.append(activation)
        modules.append(torch.nn.AdaptiveMaxPool2d((4, 4)))
        modules.append(torch.nn.Flatten())

        nb_neurons = [net_config.n_channels[-1]*4*4] + [n for n in net_config.dense_hiddens]
        #Couches complètement connectées
        for i in range(len(nb_neurons)-1):
            modules.append(torch.nn.Linear(nb_neurons[i], nb_neurons[i+1]))
            modules.append(activation)
        #Dernière couche
        modules.append(torch.nn.Linear(nb_neurons[-1], 1))
        return torch.nn.Sequential(*modules)

    @staticmethod
    def create_activation_function(activation_str: str) -> torch.nn.Module:
        if activation_str == "relu":
            return torch.nn.ReLU()
        elif activation_str == "sigmoid":
            return torch.nn.Sigmoid()
        elif activation_str == "tanh":
            return torch.nn.Tanh()
        else:
            raise ValueError("Activation function not supported")

    def compute_loss_and_mae(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        X.requires_grad = True
        mae_loss_fn = torch.nn.L1Loss()
        mse_loss_fn = torch.nn.MSELoss()
        mae_loss = mae_loss_fn(self.network(X), y)
        mse_loss = mse_loss_fn(self.network(X), y)
        return (mse_loss, mae_loss)

    def training_step(self, X_batch: torch.Tensor, y_batch: torch.Tensor):
        self.optimizer.zero_grad()
        mse_loss, mae_loss = self.compute_loss_and_mae(X_batch, y_batch)
        mse_loss.backward()
        mae_loss.backward()
        self.optimizer.step()

    def log_metrics(self, X_train: torch.Tensor, y_train: torch.Tensor,
                    X_test: torch.Tensor, y_test: torch.Tensor) -> None:
        self.network.eval()
        with torch.inference_mode():
            train_loss, train_mae = self.compute_loss_and_mae(X_train, y_train)
            test_loss, test_mae = self.compute_loss_and_mae(X_test, y_test)
        self.train_logs['train_mae'].append(train_mae.item())
        self.train_logs['test_mae'].append(test_mae.item())
        self.train_logs['train_loss'].append(train_loss.item())
        self.train_logs['test_loss'].append(test_loss.item())

    def train_loop(self, n_epochs: int):
        # Prepare train and validation data
        X_train, y_train = self.train
        X_test, y_test = self.test

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        self.log_metrics(X_train[:2000], y_train[:2000], X_test, y_test)
        for epoch in tqdm(range(n_epochs)):
            for batch in range(n_batches):
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                self.training_step(minibatchX, minibatchY)
            self.log_metrics(X_train[:2000], y_train[:2000], X_test, y_test)
        return self.train_logs

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.compute_loss_and_mae(X, y)
