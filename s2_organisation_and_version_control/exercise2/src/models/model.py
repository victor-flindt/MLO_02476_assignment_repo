from torch import nn
import torch.nn.functional as F
import torch

class MyAwesomeModel(nn.Module):
  def __init__(self):
    super(MyAwesomeModel,self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5)
    self.conv2 = nn.Conv2d(in_channels=6,out_channels=12,kernel_size=5)
    self.fc1 = nn.Linear(in_features=12*4*4,out_features=120)
    self.fc2 = nn.Linear(in_features=120,out_features=60)
    self.fc3 = nn.Linear(in_features=60,out_features=40)
    self.out = nn.Linear(in_features=40,out_features=10)
  def forward(self,x):
    #input layer
    x = x
    #first hidden layer
    x = self.conv1(x)
    x = F.relu(x)
    x = F.max_pool2d(x,kernel_size=2,stride=2)
    #second hidden layer
    x = self.conv2(x)
    x = F.relu(x)
    x = F.max_pool2d(x,kernel_size=2,stride=2)
    #third hidden layer
    x = x.reshape(-1,12*4*4)
    x = self.fc1(x)
    x = F.relu(x)
    #fourth hidden layer
    x = self.fc2(x)
    x = F.relu(x)
    #fifth hidden layer
    x = self.fc3(x)
    x = F.relu(x)
    #output layer
    x = self.out(x)
    return x