# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 12:07:36 2023

@author: Jasic
"""

import os, sys, pickle, time, warnings, re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, hidden_size=10, seq_length=24):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size,
                            num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size * seq_length, 1) # fully connected last layer
    
    def forward(self, x):
        h_0 = Variable(torch.zeros(1, x.shape[0], self.hidden_size)) # hidden state
        c_0 = Variable(torch.zeros(1, x.shape[0], self.hidden_size)) # internal state
        # Propagate input through LSTM
        output, (_, _) = self.lstm(x, (h_0, c_0)) # lstm with input, hidden, and internal state
        output = output.contiguous().view(x.shape[0], -1) # reshaping the data for Dense layer
        out = self.fc(output) # Final Output
        return out
    

df_power_sample = pd.read_csv(r'C:\Users\Jasic\OneDrive - SNU\2023-2\01 AI2\44 Codes\data\household_power_consumption_sample.csv')
print(df_power_sample)

power_sample = df_power_sample[['Elec']].to_numpy()
print(power_sample)

scaler = StandardScaler()
scaler = scaler.fit(power_sample[:-24*7])
power_sample_scaled = scaler.transform(power_sample)

X = []
y = []
for i in range(24, len(power_sample_scaled)):
    X.append(power_sample_scaled[i-24: i])
    y.append(power_sample_scaled[i])
    
    
X_train = X[:-24*7]
y_train = y[:-24*7]
X_test = X[-24*7:]
y_test = y[-24*7:]

X_train = Variable(torch.Tensor(X_train))
y_train = Variable(torch.Tensor(y_train))
X_test = Variable(torch.Tensor(X_test))
y_test = Variable(torch.Tensor(y_test))
print(X_train)
print(y_train)

torch.manual_seed(0)

# initiate lstm instance
lstm = LSTM()
criterion = torch.nn.MSELoss() # loss function
optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)

for epoch in range(1, 21):
    loss_epoch = 0
    n = 0
    for i in range(0, X_train.shape[0], 16):
        batch_X = X_train[i:i+16]
        batch_y = y_train[i:i+16]
        # caluclate the gradient, manually setting to 0
        optimizer.zero_grad()
        # forward pass
        outputs = lstm.forward(batch_X)
        # obtain the loss function
        loss = criterion(outputs, batch_y)
        loss_epoch += loss.item()
        n += 1
        # calculates the loss of the loss function
        loss.backward() 
        # improve from loss with backprop
        optimizer.step()
    # display epoch results
    loss_epoch /= n
    print(f'Epoch: {epoch}, loss: {loss_epoch}')
    
# test the model
lstm.eval()
with torch.no_grad():
    y_pred = lstm(X_test).detach().numpy().reshape(-1, 1)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(rmse)