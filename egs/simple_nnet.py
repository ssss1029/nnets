
"""
This is an implementation of a simple neural network that makes use of PyTorch's autograd 
do do backprop. We have a single hidden layer networn that should be able to fit the XOR function.

This exmaple is given in deeplearningbook.org
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDNNModel(nn.Module):
    """
    Simple DNN Model to showcase how autograd works.
    """

    def __init__(self):
        super(SimpleDNNModel, self).__init__()
        self.fc1 = nn.Linear(2, 2, bias=True)
        self.fc2 = nn.Linear(2, 1, bias=True)

        # Custom weight initialization
        # If we don't do this, the network gets stuck at non-optimal stationary points
        # (e.g. all zeros). To keep it simple, we can manually initialize the network
        # at a point close to the global optimum so that it goes toward the correct
        # stationary point.
        with torch.no_grad():
            self.fc1.weight = nn.Parameter(data=torch.tensor([
                    [1.1, 1],
                    [1, 1.1]
                ]).float())
            self.fc1.bias = nn.Parameter(data=torch.tensor([0, -1]).float())
            self.fc2.weight = nn.Parameter(data=torch.tensor([
                    [1.1, -2.1]
                ]).float())
            self.fc2.bias = nn.Parameter(data=torch.tensor(0).float())

    def forward(self, x):
        """
        Do a forward computation using the input x
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = SimpleDNNModel()

"""
Let's set up our training data so that the network learns the XOR function
Our only goal is to fit the training data perfectly.
(X, y) is the trianing data.
X contains inputs for the network
y contains true class labels.
""" 

X = torch.tensor([[0, 0],
     [1, 0],
     [0, 1],
     [1, 1]]).float()

y = torch.tensor([[0],
     [1],
     [1],
     [0]]).float()

PRINT_DETAILS = False

# Update the weights of the network
import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.01)
criterion = nn.MSELoss()

for iteration in range(1000):
    optimizer.zero_grad()
    output = net(X)
    if PRINT_DETAILS:
        print("--------- Beginning Iteration {0} --------".format(iteration))
        print("fc1.weight =", net.fc1.weight)
        print("fc1.bias =", net.fc1.bias)
        print("fc2.weight =", net.fc2.weight)
        print("fc2.bias =", net.fc2.bias)
        print("output =", output)

    loss = criterion(output, y)
    print("MSE Loss =", loss)

    loss.backward()
    optimizer.step()

