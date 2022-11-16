"""
Testing GPU and implementing a Resnet.

Learning How to integrate tensorboard with torch.
"""
import torch
from torch.utils.data import DataLoader, Subset
from torch import nn
from torchvision import datasets, transforms
from torch.nn import functional as F
from tqdm import tqdm
from typing import Tuple
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

writer = SummaryWriter("/home/sean/Documents/Work/comp_0088/labs/tensorboard/test")

def image_to_prob(model: nn.Module, images):
    """
    Return predicted probabilities from a trained network
    Args:
        model: Trained Network.
        images: The values to be tested.
    Returns:
        probs: List of output distributions
    """
    outs = model(images)
    _, preds = torch.max(output, 1)
    preds = np.squeeze(preds.to_numpy())
    return preds, [el[i].item() for i, el in zip(preds,outs)]

def probs_labels(model, dataloader):
    class_probs = []
    class_labels = []
    with torch.no_grad():
        for X, y in dataloader:
            if torch.cuda.is_available():
                X, y = X.to("cuda"), y.to("cuda")
            out = model(X)
            class_probs.append(out)
            class_labels.append(y)
    test_probs = torch.cat(class_probs)
    test_label = torch.cat(class_labels)
    return test_probs, test_label


def add_pr_curve_tensorboard(class_index, test_probs, test_label, step=0):
    truths = test_label == class_index
    probs = test_probs[:, class_index]
    classes = [i for i in range(10)]
    writer.add_pr_curve(str(classes[class_index]),
                        truths,
                        probs,
                        global_step=step)
    writer.close()


def train_epoch(model, dataloader, loss_function, optimiser):
    """
    Train a model on a single epoch of data.

    # Arguments
        model: a pytorch model (eg one of the above nn.Module subclasses)
        dataloader: a pytorch dataloader that will iterate over the dataset
            in batches
        loss_function: a pytorch loss function tensor
        optimiser: optimiser to use for training

    # Returns:
        loss: mean loss over the whole epoch
        accuracy: mean prediction accuracy over the epoch
    """
    assert isinstance(model, nn.Module)
    assert isinstance(dataloader, DataLoader)
    assert isinstance(optimiser, torch.optim.Optimizer)
    loss_train = 0
    acc_train = 0
    for i, (X, y) in enumerate(dataloader):
        X, y = X.to("cuda"), y.to("cuda")
        y_hat = model(X)
        y_hot = torch.zeros_like(y_hat, device="cuda")
        y_hot = y_hot.scatter_(1, y.unsqueeze(1), 1.0)
        # compute loss
        accuracy = (torch.argmax(y_hat, 1) == y).sum() / y.shape[0]
        loss = loss_function(y_hat, y_hot)
        loss_train += loss.item()
        acc_train += accuracy.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    return loss_train / len(dataloader), acc_train / len(dataloader)


def test_epoch(model, dataloader, loss_function):
    """
    Evaluate a model on a dataset.

    # Arguments
        model: a pytorch model (eg one of the above nn.Module subclasses)
        dataloader: a pytorch dataloader that will iterate over the dataset
            in batches
        loss_function: a pytorch loss function tensor

    # Returns:
        loss: mean loss over the whole epoch
        accuracy: mean prediction accuracy over the epoch
    """
    assert isinstance(model, nn.Module)
    assert isinstance(dataloader, DataLoader)
    loss_test = 0
    acc_test = 0
    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):
            X, y = X.to("cuda"), y.to("cuda")
            y_hat = model(X)
            y_hot = torch.zeros_like(y_hat, device="cuda")
            y_hot.scatter_(1, y.unsqueeze(1), 1.0)
            accuracy = (torch.argmax(y_hat, dim=1) == y).sum() / y.shape[0]
            acc_test += accuracy.item()
            loss = loss_function(y_hat, y)
            loss_test += loss.item()
    return loss_test / len(dataloader), acc_test / len(dataloader)


class ResBlock(nn.Module):
    """
    Implement classic resnet architecture building block.

    """

    def __init__(self, channels):
        super().__init__()
        # Define the Parameterized Layers
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(num_features=channels)
        # Define the initialization weights
        torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        torch.nn.init.constant_(self.batch_norm.weight, 0.5)
        torch.nn.init.zeros_(self.batch_norm.bias)

    def forward(self, X):
        """
        Forward pass on block.
        :param X:
        :return:
        """
        out = self.conv(X)
        out = self.batch_norm(out)
        out = torch.relu(out)
        return out + X


class ResNet(nn.Module):
    """
    Implements Classic ResNet.
    """

    def __init__(self, input_shape: Tuple[int], output_classes: int,n_block: int=10, block_chan:int = 64):
        super().__init__()
        self.chan_0 = input_shape[0]
        self.output_classes = output_classes
        self.block_chan = block_chan
        self.conv1 = nn.Conv2d(self.chan_0, self.block_chan, kernel_size=3, padding=1)
        self.resblock = nn.Sequential(*(n_block * [ResBlock(self.block_chan)]))
        self.l1 = nn.Linear(input_shape[2]//4 * input_shape[1]//4 * self.block_chan, self.output_classes*4)
        self.l2 = nn.Linear(self.output_classes*4, self.output_classes)
    def forward(self, X):
        out = F.max_pool2d(F.relu(self.conv1(X)), 2)
        out = self.resblock(out)
        out = F.max_pool2d(out, 2)
        out = nn.Flatten()(out)
        out = F.relu(self.l1(out))
        out = F.softmax(self.l2(out), dim=-1)
        return out


if __name__ == "__main__":
    print("Using Cuda:", torch.cuda.is_available())
    train = datasets.FashionMNIST(
        "FMNIST", download=True, train=True, transform=transforms.ToTensor()
    )
    val = datasets.FashionMNIST(
        "FMNIST", download=True, train=False, transform=transforms.ToTensor()
    )

    train_data_loader = DataLoader(train, 64, shuffle=True, num_workers=10)
    val_data_loader = DataLoader(val, 64, shuffle=True, num_workers=10)
    EPOCH = 20
    x_in, _ = train[0]
    classes = 10
    model = ResNet(x_in.shape, 10)
    model.to("cuda")
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    acc = None
    loss = None
    for epoch in (tq := tqdm(range(EPOCH), leave=False)):
        #Train
        loss, acc = train_epoch(model, train_data_loader, loss_fn, opt)
        train_loss.append(loss)
        train_acc.append(acc)
        #Validation
        loss_val, acc_val = test_epoch(model, val_data_loader, loss_fn)
        val_loss.append(loss_val)
        val_acc.append(acc_val)
        writer.add_scalar("Training Loss", loss, int(epoch))
        writer.add_scalar("Training Accuracy", acc, int(epoch))
        writer.add_scalar("Validation Loss", loss_val, int(epoch))
        writer.add_scalar("Validation Accuracy", acc_val, int(epoch))
    ## Precision Recall Curve

    t_prob, t_label = probs_labels(model, val_data_loader)
    for i in range(10):
        add_pr_curve_tensorboard(i, t_prob, t_label)