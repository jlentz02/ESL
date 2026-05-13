#Simple linear neural network using Pytorch

#Imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from General_methods import load_data, tt_split, normalize_data, test

#Make simple linear NN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #IDK what I am doing so I am just gonna try some stuff out
        #Layer 1: 23 -> 23
        self.fc1 = nn.Linear(23, 64)
        #Layer 2, hidden layer: 23 -> 23
        self.fc2 = nn.Linear(64, 32)
        #Layer 3: 23 -> 1
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
    

#Load data
data_X, data_Y, columns = load_data("UCI_Credit_Card.csv", ysplit = False)
trainval_X, trainval_Y, test_X, test_Y = tt_split(data_X, data_Y)
train_X, train_Y, val_X, val_Y = tt_split(trainval_X, trainval_Y, .75)
#Normalize data
nt_X, mean, std = normalize_data(train_X)
#Batch data for training
train_data = TensorDataset(nt_X, train_Y)
batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

#Define optimizer and loss function
loss_fn = nn.BCELoss()
model = Net()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
n = len(train_X)
#Main loop
for epoch in range(100):
    epoch_loss = 0.0
    epoch_acc = 0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(x_batch)
        #Niche fix since we have binary output
        #Net returns a tensor, but our y value isn't a tensor
        with torch.no_grad():
            for i in range(len(outputs)):
                if torch.round(outputs[i][0])== y_batch[i]:
                    epoch_acc+=1
        loss = loss_fn(outputs, y_batch.view(-1, 1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss
    if epoch%10 == 0 or epoch == 99:
        print(f"Loss on epoch {epoch}: {torch.round(batch_size*epoch_loss/n, decimals=4)}")
        print(f"Error on epoch {epoch}: {round(1 - epoch_acc/n,4)}")
    epoch_loss = 0
    epoch_acc = 0

print("---------------------")
val_pred = model((val_X - mean)/std)
val_error = test(val_Y, val_pred)
print(f"Validation error: {val_error}")