"""=============================================================================
PyTorch implementation of LeNet-5. See:

    http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
============================================================================="""

import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------------------

class LeNet5AE(nn.Module):

    def __init__(self, cfg):
        """Initialize LeNet5.
        """
        super(LeNet5AE, self).__init__()

        self.nc = cfg.N_CHANNELS
        self.w  = cfg.IMG_SIZE

        # In-channels, out-channels, kernel size. See `forward()` for
        # dimensionality analysis.
        self.conv1 = nn.Conv2d(self.nc, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, cfg.IMG_EMBED_DIM)

        self.fc4 = nn.Linear(cfg.IMG_EMBED_DIM, 84)
        self.fc5 = nn.Linear(84, self.nc * self.w * self.w)

# ------------------------------------------------------------------------------

    def encode(self, x):
        x = F.pad(x, (2, 2, 2, 2))
        y = F.relu(self.conv1(x))  # nc x 32 x 32 ---> 6  x 28 x 28
        y = F.max_pool2d(y, 2)     # 6  x 28 x 28 ---> 6  x 14 x 14
        y = F.relu(self.conv2(y))  # 6  x 14 x 14 ---> 16 x 10 x 10
        y = F.max_pool2d(y, 2)     # 16 x 10 x 10 ---> 16 x 5  x 5
        y = y.view(y.size(0), -1)  # 16 x 5  x 5  ---> 400
        y = F.relu(self.fc1(y))    # 400          ---> 120
        y = F.relu(self.fc2(y))    # 120          ---> 84
        y = self.fc3(y)            # 84           ---> k
        return y

# ------------------------------------------------------------------------------

    def decode(self, z):
        y = F.relu(self.fc4(z))
        y = F.relu(self.fc5(y))
        y = y.view(-1, self.nc, self.w, self.w)
        return y

# ------------------------------------------------------------------------------

    def forward(self, x):
        """Perform forward pass on neural network.
        """
        x = self.encode(x)
        x = self.decode(x)
        return x
