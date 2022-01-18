import argparse
import sys
import torch.nn.functional as F
from torch import nn, optim
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import torch

from data import mnist
from model import MyAwesomeModel
#from mikkel_dataloader import mnist


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
        #model = MyAwesomeModel()
        train, _ = mnist()
        trainloader = DataLoader(train, batch_size=64, shuffle=True)

        model = nn.Sequential(nn.Linear(784, 128),
                            nn.ReLU(),
                            nn.Linear(128, 64),
                            nn.ReLU(),
                            nn.Linear(64, 10),
                            nn.LogSoftmax(dim=1))

        criterion = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.003)

        epochs = 5
        for e in range(epochs):
            running_loss = 0
            for images, labels in trainloader:
                # Flatten MNIST images into a 784 long vector
                images = images.view(images.shape[0], -1)
            
                # TODO: Training pass
                optimizer.zero_grad()
                
                output = model(images)
                loss = criterion(output, labels)
                
                #This is where the model learns by backpropagating
                loss.backward()
                
                #And optimizes its weights here
                optimizer.step()
                
                running_loss += loss.item()
            else:
                print(f"Training loss: {running_loss/len(trainloader)}")

    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        model = torch.load(args.load_model_from)
        _, test_set = mnist()

        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for images, labels in test_set:
                # calculate outputs by running images through the network
                outputs = model(images.float().unsqueeze(dim=2).unsqueeze(dim=3))
                # the class with the highest energy is what we choose as prediction
                print(test_set)
                _, predicted = torch.max(outputs.data)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 5000 test images: %d %%' % (
            100 * correct / total))



if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    