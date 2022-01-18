import argparse
import sys
from datetime import date, datetime
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision
from torch import nn, optim
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os
from os import path
import pathlib
import torch

from data import mnist
from model import MyAwesomeModel
#from mikkel_dataloader import mnist
def save_model(model):

    #base_dir = str(pathlib.Path(__file__).parent.resolve())+"/"
    now = datetime.now()
    dirname = now.strftime(f"%Y-%m-%d")
    if path.exists(dirname):
        minut = now.strftime("%H-%M-%S")
        os.makedirs("models/", exist_ok=True)
        torch.save(model.state_dict(), f"models/trained_model{minut}.pt")
    else:
        os.makedirs(dirname)
        minut = now.strftime("%H-%M-%S")
        os.makedirs("models/", exist_ok=True)
        torch.save(model.state_dict(), f"models/trained_model{minut}.pt")
    print(f'Saved model in as model.pt')
def save_picture(arr1,arr2):
    #base_dir = str(pathlib.Path(__file__).parent.resolve())+"/"
    now = datetime.now()
    dirname = now.strftime(f"%Y-%m-%d")
    if path.exists(dirname):
        minut = now.strftime("%H-%M-%S")
        plt.plot(arr1,arr2)
        plt.xlabel("Epochs")
        plt.ylabel("Traning loss")
        plt.grid()
        plt.savefig(f"TL{minut}.png")
    else:
        os.makedirs(dirname)
        minut = now.strftime("%H-%M-%S")
        plt.plot(arr1,arr2)
        plt.xlabel("Epochs")
        plt.ylabel("Traning loss")
        plt.grid()
        plt.savefig(f"TL{minut}.png")
    print(f'Saved figure as TL{minut}.pt')


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here
        #datatset from pytorch, my dataworker stopped working propably. 

        # train_set = torchvision.datasets.FashionMNIST(
        # root = './data/FashionMNIST',
        # download = True,
        # train = True,
        # transform = transforms.Compose([
        # transforms.ToTensor(),
        # ])
        # )

        train_set = mnist()
        train_loader = DataLoader(train_set, batch_size=20)


        #defining parameters
        model = MyAwesomeModel()
        learning_rate = 0.005
        optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        epochs = 2

        #training loop
        losses = []
        epoch_arr = []
        for i in range(epochs):
            for j,(images,targets) in enumerate(train_loader):
                
                #making predictions
                y_pred = model(images)
            
                #calculating loss
                loss = criterion(y_pred,targets.reshape(-1))
                loss_tem = loss.detach().numpy()
                #backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if i>10:
                optimizer.lr = 0.0005
            print(loss)
            losses.append(loss_tem)
            epoch_arr.append(i)

        save_model(model)
        #save_picture(epoch_arr,losses)

if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    