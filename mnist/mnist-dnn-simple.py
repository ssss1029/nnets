"""
A simple fully-connected DNN built to classify images from the MNIST dataset
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
import time
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Global:
    train_file = "data/train.csv"
    test_file = "data/test.csv"
    use_gpu = True
    epochs = 500
    batch_size = 7000

class SimpleMNISTDNNModel(nn.Module):

    def __init__(self):
        super(SimpleMNISTDNNModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 700)
        self.fc2 = nn.Linear(700, 700)
        self.fc3 = nn.Linear(700, 200)
        self.fc4 = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x
    

def main():
    # Read the input data and construct the input matrix
    X = []
    y = []
    with open(Global.train_file, 'r', newline='') as train_file:
        csvreader = csv.reader(train_file, delimiter=",")
        next(csvreader) # Skip the first row. These are just column labels
        for row in csvreader:
            label = int(row[0])
            feats = [int(e) for e in row[1:]]
            assert(len(feats)) == 28 * 28
            X.append(feats)
            y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    num_train = len(y)
    
    assert X.shape[0] == len(y)
    assert num_train % Global.batch_size == 0 

    print("Finished loading training data")

    # Initialize a neural network
    if Global.use_gpu:
        net = SimpleMNISTDNNModel()
        net.cuda()
    else:
        net = SimpleMNISTDNNModel()

    # Train the neural network using SGD
    optimizer = optim.SGD(net.parameters(), lr=0.001)  
    loss_fn = nn.CrossEntropyLoss()

    idx = np.arange(num_train)
    curr_X = X
    curr_y = y
    starttime = time.time()
    epoch_scores = []
    for epoch in range(Global.epochs):
        
        np.random.shuffle(idx)
        curr_X = X[idx]
        curr_y = y[idx]
        
        for batch_num in range(num_train // Global.batch_size):
            curr_input = curr_X[Global.batch_size * batch_num : Global.batch_size * (batch_num + 1)]
            curr_target = curr_y[Global.batch_size * batch_num : Global.batch_size * (batch_num + 1)]
            
            curr_input = torch.tensor(curr_input, dtype=torch.float)
            curr_target = torch.tensor(curr_target, dtype=torch.long)

            if Global.use_gpu:
                curr_input = curr_input.cuda()
                curr_target = curr_target.cuda()

            optimizer.zero_grad()
            output = net(curr_input)
            loss = loss_fn(output, curr_target)
            loss.backward()
            optimizer.step()

        
        # After-epoch evaluation:
        with torch.no_grad():
            curr_X_tensor = torch.tensor(curr_X, dtype=torch.float)
            curr_y_tensor = torch.tensor(curr_y, dtype=torch.long)

            if Global.use_gpu:
                curr_X_tensor = curr_X_tensor.cuda()
                curr_y_tensor = curr_y_tensor.cuda()

            output = net(curr_X_tensor)
            loss = loss_fn(output, curr_y_tensor)

            print("Epoch: {:03}, Loss = {:07}".format(epoch, loss))
            epoch_scores.append(loss)

    secs = time.time() - starttime
    print(secs)

    plt.plot([float(e) for e in epoch_scores])
    plt.show()

main()

