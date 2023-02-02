import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/uclaacmai/leaf-us-alone


class StartingNetwork(nn.Module):
  def __init__(self):
    # Call nn.Module's constructor--don't forget this
    super().__init__()

    # Define layers
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(28 * 28, 256)
    self.fc2 = nn.Linear(256, 128)
    self.fc3 = nn.Linear(128, 10)

  def forward(self, x):
    # Forward propagation
    x = self.flatten(x)
    x = self.fc1(x)
    x = F.relu(x)

    x = self.fc2(x)
    x = F.relu(x)

    x = self.fc3(x)

    # No activation function at the end
    # nn.CrossEntropyLoss takes care of it for us

    return x



class ConvNet(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self):
        super().__init__()

        # TODO: Change the dimensions of layers to match that of our dataset (inputted)



        # Conv2d expects the following arguments
        #   - C, the number of channels in the input
        #   - C', the number of channels in the output
        #   - The filter size (called a kernel size in the documentation)
        #     Below, we specify 5, so our filters will be of size 5x5.
        #   - The amount of padding (default = 0)
        self.conv1 = nn.Conv2d(1, 4, kernel_size=5, padding=2) # notice how we use padding to prevent dimension reduction

        # Pooling layer takes two arguments
        #   - Filter size (in this case, 2x2)
        #   - Stride
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(8 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

        # Original Code Here
        # self.flatten = nn.Flatten()
        # self.fc = nn.Linear(224 * 224 * 3, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Comments below give the shape of x
        # n is batch size

        # (n, 1, 28, 28)
        x = self.conv1(x)
        x = F.relu(x)
        # (n, 4, 28, 28)
        x = self.pool(x)
        # (n, 4, 14, 14)
        x = self.conv2(x)
        x = F.relu(x)
        # (n, 8, 14, 14)
        x = self.pool(x)
        # (n, 8, 7, 7)
        x = torch.reshape(x, (-1, 8 * 7 * 7))
        # (n, 8 * 7 * 7)
        x = self.fc1(x)
        x = F.relu(x)
        # (n, 256)
        x = self.fc2(x)
        x = F.relu(x)
        # (n, 128)
        x = self.fc3(x)
        # (n, 10)
        return x




"""
Code from the colab notebook on CV:
class ConvNet(nn.Module):
  def __init__(self):
    # Call nn.Module's constructor--don't forget this
    super().__init__()

    \"""
    Define layers
    \"""
    # Explanation of arguments
    # Remember a Convolution layer will take some input volume HxWxC
    # (H = height, W = width, and C = channels) and map it to some output
    # volume H'xW'xC'.
    #
    # Conv2d expects the following arguments
    #   - C, the number of channels in the input
    #   - C', the number of channels in the output
    #   - The filter size (called a kernel size in the documentation)
    #     Below, we specify 5, so our filters will be of size 5x5.
    #   - The amount of padding (default = 0)
    self.conv1 = nn.Conv2d(1, 4, kernel_size=5, padding=2) #notice how we use padding to prevent dimension reduction
    self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)

    # Pooling layer takes two arguments
    #   - Filter size (in this case, 2x2)
    #   - Stride
    self.pool = nn.MaxPool2d(2, 2)

    self.fc1 = nn.Linear(8 * 7 * 7, 256)
    self.fc2 = nn.Linear(256, 128)
    self.fc3 = nn.Linear(128, 10)

  def forward(self, x):
    # Comments below give the shape of x
    # n is batch size

    # (n, 1, 28, 28)
    x = self.conv1(x)
    x = F.relu(x)
    # (n, 4, 28, 28)
    x = self.pool(x)
    # (n, 4, 14, 14)
    x = self.conv2(x)
    x = F.relu(x)
    # (n, 8, 14, 14)
    x = self.pool(x)
    # (n, 8, 7, 7)
    x = torch.reshape(x, (-1, 8 * 7 * 7))
    # (n, 8 * 7 * 7)
    x = self.fc1(x)
    x = F.relu(x)
    # (n, 256)
    x = self.fc2(x)
    x = F.relu(x)
    # (n, 128)
    x = self.fc3(x)
    # (n, 10)
    return x

"""