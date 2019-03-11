#!/usr/bin python
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from itertools import product
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import os
from torchvision.datasets import ImageFolder, DatasetFolder
import torch.utils.data

import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def image_loader(image_name, imsize = 240):
    """load image, returns cuda tensor"""
    image = Image.open(image_name).convert('RGB')

    loader = transforms.Compose([transforms.Scale(imsize), transforms.CenterCrop(imsize), transforms.ToTensor(), normalize])
    image = loader(image).float()
    image = Variable(image, requires_grad=False)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image#.to_device(device)  #assumes that you're using GPU

class AttRNN(nn.Module):
    def __init__(self, hidden_size=4, num_rnn_layers=5):
        super(AttRNN, self).__init__()
        self.hidden_size = hidden_size
        
        vgg19 = models.vgg19(pretrained=True)
        self.earlyLayers = nn.Sequential(*list(vgg19.children())[0][:28]).to(device)

        for param in self.earlyLayers.parameters():
            param.requires_grad = False

        self.rnn = nn.LSTM(115200, self.hidden_size, num_rnn_layers, dropout=.05).to(device)
        self.out = nn.Linear(hidden_size, 4).to(device)

    def forward(self, inputs, hidden=None):
        inp_ = self.earlyLayers(inputs)
        # Unravel the earlyLayers output before sending into the RNN
        inp_ = inp_.view(inp_.size(0),-1).unsqueeze(0)
        
        output, hidden = self.rnn(inp_, hidden)
        output = self.out(output.squeeze(1))
        return output, hidden

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, num_classes)) 

    def forward(self, inp):
        out = self.mlp(inp)
        return out

class AttMLP(nn.Module):
    def __init__(self, hidden_size=128, num_classes=4):
        super(AttMLP, self).__init__()
        self.hidden_size = hidden_size

        vgg19 = models.vgg19(pretrained=True)
        self.earlyLayers = nn.Sequential(*list(vgg19.children())[0][:28]).to(device)
        for param in self.earlyLayers.parameters():
            param.requires_grad = False

        self.mlp = MLP(115200+num_classes, self.hidden_size, num_classes).to(device)
        for param in self.mlp.parameters():
            param.requires_grad = True
        #self.out = nn.Linear(hidden_size, 4).to(device)

    def forward(self, inputs, query):
        inp_ = self.earlyLayers(inputs)
        # Unravel the earlyLayers output before sending into the RNN
        inp_ = inp_.view(inp_.size(0),-1).unsqueeze(0)
        
        # Concatenate the input vector with the query
        cat = torch.cat((inp_, query), dim=2)
        
        output = self.mlp(cat)
        #output = self.out(output.squeeze(1))
        return output
    
def train_AttMLP(savedir = '/scratch/users/akshayj/log002', n_epochs=50):
    hidden_size = 128
    num_classes = 4
    batch_size = 50
    
    # make savedir if it does not exist
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Load data
    stimdir = '/home/users/akshayj/att_net/attention/imagenet_stimuli/'
    train_dataset = ImageFolder(stimdir, loader=image_loader)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create Model
    model = AttMLP(hidden_size, num_classes).to(device)
    loss_func = nn.MSELoss()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.01)

    train_losses = np.zeros(n_epochs) # For plotting

    for epoch in range(n_epochs):
        test_true = []; test_pred = []; test_losses = [] 

        
        for step, (x,y) in enumerate(train_data_loader):
            b_x = Variable(torch.squeeze(x)).cuda()
            
            # Turn Y vector into one-hot
            b_y = torch.fmod(y,4).unsqueeze(1).to(device)
            y_1hot = torch.FloatTensor(y.shape[0], 4).zero_().to(device)
            y_1hot.scatter_(1, b_y, 1)

            # Extract hidden unit initialization and turn into one-hot
            hid = torch.div(y,4).unsqueeze(1).to(device);
            hid_1hot = torch.FloatTensor(hid.shape[0], 4).zero_().to(device)    
            hid_1hot.scatter_(1, hid, 1)
            hid_1hot = hid_1hot.unsqueeze(0)
            
            # Get training loss.
            output = model(b_x, hid_1hot)
            #m = nn.Sigmoid()
            loss = loss_func(output, y_1hot)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses[epoch] += loss.item()
            if step % 10 == 0:
                print 'Step {}: Training Loss = {}'.format(step, loss.item())

        print '---Epoch {}: Loss = {} ---'.format(epoch, train_losses[epoch])
        save_dict = {'epoch': epoch, 'train_loss':train_losses}
        np.save(savedir+'/epoch{}.npy'.format(epoch), save_dict);
        
        