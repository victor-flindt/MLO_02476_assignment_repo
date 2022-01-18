
import torch
from torch.functional import Tensor
import torchvision
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
from ptflops import get_model_complexity_info


def main():
    MobileNet   = torchvision.models.mobilenet_v3_large(pretrained=True, width_mult=1.0,  reduced_tail=False, dilated=False)
    ResNet      = torchvision.models.resnet152(pretrained=True)
    tensor = torch.tensor([1, 2, 3, 4])

    output = ResNet(torch.unsqueeze(tensor,0))
    # running complexity analysis on the model Mobilenet
    macs, params = get_model_complexity_info(MobileNet,
                                            (3, 229, 229),
                                            as_strings=True,
                                            print_per_layer_stat=True,
                                            verbose=True) 


    # running complexity analysis on the model Resnet
    macs1, params1 = get_model_complexity_info(ResNet,
                                            (3, 229, 229),
                                            as_strings=True,
                                            print_per_layer_stat=True,
                                            verbose=True) 
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    print('{:<30}  {:<8}'.format('Computational complexity: ', macs1))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params1))

main()







main()