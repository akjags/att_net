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
m = nn.Upsample(scale_factor=10, mode='nearest')

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

        self.mlp = MLP(115200+4, self.hidden_size, num_classes).to(device)
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

class AttNet(nn.Module):
    def __init__(self, hidden_size=128, num_classes=576):
        super(AttNet, self).__init__()
        self.hidden_size = hidden_size

        vgg19 = models.vgg19(pretrained=True)
        self.pool4 = nn.Sequential(*list(vgg19.children())[0][:28]).to(device)
        for param in self.pool4.parameters():
            param.requires_grad = False

        self.mlp = MLP(115200+4, self.hidden_size, num_classes).to(device)
        for param in self.mlp.parameters():
            param.requires_grad = True
        
        self.conv1 = nn.Sequential(*list(vgg19.children())[0][:1]).to(device)
        self.convRest = nn.Sequential(*list(vgg19.children())[0][1:]).to(device)
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.convRest.parameters():
            param.requires_grad = False
        
        self.lin = nn.Linear(512*7*7, 4)

    def forward(self, inputs, query):
        # First get the pool4 output
        inp_ = self.pool4(inputs)
        # Unravel the earlyLayers output before sending into the RNN
        inp_ = inp_.view(inp_.size(0),-1).unsqueeze(0)
        # Concatenate the input vector with the query
        cat = torch.cat((inp_, query), dim=2)
        
        # Get the output of the Multilayer Perceptron (gain)
        mlp_out = self.mlp(cat) # 1 x 240
        mlp_out = mlp_out.view(1,-1,24,24)
        mlp_upsmpl = torch.squeeze(m(mlp_out))
        gain_map = torch.stack((mlp_upsmpl,)*64, dim=0).transpose(0,1)
        
        # Multiply by the output of conv1 
        conv1_out = self.conv1(inputs) # batch_size x 64 x 240 x 240
        gain_prod = torch.mul(conv1_out, gain_map)
        
        # Pass the multiplicatively enhanced thing through the 
        convRest_out = self.convRest(gain_prod)
        convRest_out = convRest_out.view(convRest_out.size(0), -1)
        output = self.lin(convRest_out)
        #print output.shape
        return output

class ImageFolderEX(ImageFolder):
    def __getitem__(self, index):
        path, label = self.imgs[index]
        try:
            img = self.loader(os.path.join(self.root, path))
        except:
            img, label = None, None
        return [img, label]

def train_AttNet(test_data, savedir = '/scratch/users/akshayj/log002', n_epochs=10):
    hidden_size = 128
    num_classes = 576
    batch_size = 50
    test_freq = 100
    
    # make savedir if it does not exist
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Load data
    stimdir = '/scratch/users/akshayj/att_net_stimuli/'
    train_dataset = ImageFolderEX(stimdir, loader=image_loader)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Get test data
    x_test, y_test, query_test = test_data
    
    # Create Model
    model = AttNet(hidden_size).to(device)
    loss_func = nn.MSELoss()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.01)

    train_losses = np.zeros((n_epochs, len(train_data_loader))) # For plotting
    test_losses  = np.zeros((n_epochs, len(train_data_loader)))
    for epoch in range(n_epochs):
        
        for step, (x,y) in enumerate(train_data_loader):
            if x is None:
                print 'IO Error caught: Skipping image on step {}'.format(step)
                continue
            b_x = Variable(torch.squeeze(x)).to(device)
            
            # Turn Y vector into one-hot
            b_y = torch.fmod(y,4).unsqueeze(1).to(device)
            y_1hot = torch.FloatTensor(y.shape[0], 4).zero_().to(device)
            y_1hot.scatter_(1, b_y, 1)
            del b_y

            # Extract hidden unit initialization and turn into one-hot
            hid = torch.div(y,4).unsqueeze(1).to(device);
            hid_1hot = torch.FloatTensor(hid.shape[0], 4).zero_().to(device)    
            hid_1hot.scatter_(1, hid, 1)
            hid_1hot = hid_1hot.unsqueeze(0)
            del hid
            
            # Get training loss.
            optimizer.zero_grad()
            output = model(b_x, hid_1hot)
            loss = loss_func(output, y_1hot)

            # Propagate
            loss.backward()
            optimizer.step()

            # Save training loss
            train_losses[epoch, step] = loss.item()
            torch.cuda.empty_cache()
            if step % test_freq == 0:
                # Get test loss.
                with torch.no_grad():
                    output = model(x_test, query_test)
                    test_loss = loss_func(output, y_test)
                    #print epoch, step, test_losses, test_loss.item()

                test_losses[epoch, step] = test_loss.item()
                save_dict = {'epoch': epoch, 'step': step, 'train_loss':train_losses, 'test_loss': test_losses}
                np.save(savedir+'/epoch{}_step{}.npy'.format(epoch, step), save_dict);
                print '---Epoch {}, Step {}: Train Loss = {}; Test Loss = {} ---'.format(epoch, step, 
                                                               train_losses[epoch,step], test_losses[epoch,step])

            
        