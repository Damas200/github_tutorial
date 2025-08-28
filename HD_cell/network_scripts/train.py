import numpy as np
import torch
import torch.nn as nn
import SAN

# train simplicial convolutional network or SARNN; saves learning rates and loss function values to files
def train(model, device, data_loader, optimizer, criterion, scheduler, epochs):
    '''
    Inputs:
        model:       neural network being trained (SARNN or SCNN)
        device:      device on which data and network are stored and using for computation
        data_loader: torch data loader holding all training data
        optimizer:   optimizer being used on network parameters
        criterion:   loss function
        scheduler:   learning rate scheduler
        epochs:      number of epochs for training
    '''
    model.train()
    losses = []

    for step in range(epochs):
        losses_tmp = []
        for i in data_loader:
            idx, sample, label = i
            label = label.view(-1, 1).float().to(device)

            if isinstance(model, SAN.SAN_RNN):
                # SARNN expects [z0, z1, z2, L0, L1_d, L1_u, L2, B1, B2]
                z0, z1, z2, L0, L1_d, L1_u, L2, B1, B2 = sample
                output = model([z0.float(), z1.float(), z2.float(), L0.float(), L1_d.float(), L1_u.float(), L2.float(), B1, B2]).to(device)
            else:
                # SCNN or other models expect sample directly
                output = model(sample.float()).to(device)

            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_tmp.append(loss.cpu().detach().numpy())
        epoch_loss = sum(losses_tmp) / len(losses_tmp)
        print('Loss at step', step+1, ':', epoch_loss)
        losses.append(epoch_loss)

        scheduler.step()

# train FFNN; saves learning rates and loss function values to files
def trainFFNN(model, device, data_loader, optimizer, criterion, epochs):
    '''
    Inputs:
        model:       neural network being trained
        device:      device on which data and network are stored and using for computation
        data_loader: torch data loader holding all training data
        optimizer:   optimizer being used on network parameters
        criterion:   loss function
        epochs:      number of epochs for training
    '''
    model.train()
    losses = []

    for step in range(epochs):
        losses_tmp = []
        for i in data_loader:
            idx, sample, label = i
            label = label.to(device)
            output = model(sample.float()).to(device)
            loss = criterion(output, label.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_tmp.append(loss.cpu().detach().numpy())
            epoch_loss = sum(losses_tmp) / len(losses_tmp)
        print('Loss at step', step+1, ':', epoch_loss)
        losses.append(epoch_loss)

# train RNN; saves learning rates and loss function values to files
def trainRNN(model, device, data_loader, optimizer, criterion, scheduler, epochs):
    '''
    Inputs:
        model:       neural network being trained
        device:      device on which data and network are stored and using for computation
        data_loader: torch data loader holding all training data
        optimizer:   optimizer being used on network parameters
        criterion:   loss function
        scheduler:   learning rate scheduler
        epochs:      number of epochs for training
    '''
    model.train()
    losses = []

    for step in range(epochs):
        losses_tmp = []
        for i in data_loader:
            idx, sample, label = i
            label = label.view(-1, 1).to(device)
            output = model(sample.float()).to(device)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_tmp.append(loss.cpu().detach().numpy())
            epoch_loss = sum(losses_tmp) / len(losses_tmp)
        print('Loss at step', step+1, ':', epoch_loss)
        losses.append(epoch_loss)
        scheduler.step()

# compute median absolute error and average absolute error 
def MAE(output, label):
    '''
    Inputs:
        output: network output across all time bins
        label: ground truth HD angles across all time bins
    Returns:
        mae: median absolute error
        aae: average absolute error
    '''
    diff = np.abs(output - label)
    diff[diff > 180.0] = np.abs(360 - diff[diff > 180.0])
    return np.median(diff), np.mean(diff)