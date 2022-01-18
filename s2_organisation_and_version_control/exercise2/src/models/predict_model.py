import argparse
import torch.nn.functional as F
from torch import nn, optim
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch
import sys
import glob
import matplotlib.pyplot as plt
from PIL import Image
from torchmetrics.classification import Accuracy
import numpy as np
from data import mnist
from model import MyAwesomeModel
import helper

def load_data():
    #_, test_data = mnist()
    test_set = torchvision.datasets.FashionMNIST(
    root = './data/FashionMNIST',
    download = True,
    train = False,
    transform = transforms.Compose([
    transforms.ToTensor(),
    ])
    )
    return test_set

def predict():
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('--model', default="")
    #parser.add_argument('--data_path', default="")
    # add any additional argument that you want

    args = parser.parse_args()

    ## Loading in entire model.
    model_dir = "C:/Users/victo/OneDrive - Danmarks Tekniske Universitet/skole/9.semester/Machine Learning Operations 02476/dtu_mlops/s2_organisation_and_version_control/exercise2/models/"
    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_dir+args.model))
    model.eval()
    
    # model = torch.load(model_dir+args.model)
    # model.eval()
    learning_rate = 0.005
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    epochs = 5

    test_data = load_data()
    testloader = DataLoader(test_data, batch_size=1, shuffle=True) 
    
    correct = 0
    total = 0
    tot_test_loss = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # claculates the test loss of the system 
            loss = criterion(outputs,labels)
            tot_test_loss += loss.item()
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            ## checks predicted labels vs actual labels for accuracy measure.
            correct += (predicted == labels).sum().item()

    test_loss = tot_test_loss / len(test_data)
    print(f'Test loss of the network: {test_loss}')
    print('Accuracy of the network: %d %%' % (
        100 * correct / total))

    # TODO: Calculate the class probabilities (softmax) for img
    # comment in for showign predictiong with picture. 
    # with torch.no_grad():
    #     logps = model(img)
    # ps = torch.exp(logps)
    # print(logps)
    # print(ps)
    # helper.view_classify(img.view(1, 28, 28), ps)


if __name__ == '__main__':
    predict()
        