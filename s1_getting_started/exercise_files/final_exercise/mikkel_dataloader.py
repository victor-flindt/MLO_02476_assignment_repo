import torch
import numpy as np
from glob import glob

def mnist():
    # exchange with the corrupted mnist dataset
    # train = torch.randn(50000, 784)
    # test = torch.randn(10000, 784) 
    direct = "C:/Users/victo/OneDrive - Danmarks Tekniske Universitet/skole/9.semester/Machine Learning Operations 02476/dtu_mlops/s1_getting_started/exercise_files/final_exercise/corruptmnist/"

    test = dict()
    train_images = []
    train_labels = []

    train0 = np.load(direct + "train_0.npz")
    train1 = np.load(direct + "train_1.npz")
    train2 = np.load(direct + "train_2.npz")
    train3 = np.load(direct + "train_3.npz")
    train4 = np.load(direct + "train_4.npz")
    
    train0_images = torch.tensor(train0['images'])
    train1_images = torch.tensor(train1['images'])
    train2_images = torch.tensor(train2['images'])
    train3_images = torch.tensor(train3['images'])
    train4_images = torch.tensor(train4['images'])
    
    train0_labels = torch.tensor(train0['labels'])
    train1_labels = torch.tensor(train1['labels'])
    train2_labels = torch.tensor(train2['labels'])
    train3_labels = torch.tensor(train3['labels'])
    train4_labels = torch.tensor(train4['labels'])

    test = np.load(direct + "test.npz")
    test_images = torch.tensor(test['images'])
    test_labels = torch.tensor(test['labels'])

    train_images = torch.cat((train0_images, train1_images, train2_images, train3_images, train4_images))
    train_labels = torch.cat((train0_labels, train1_labels, train2_labels, train3_labels, train4_labels))
    print("len of train_images:",len(train_images))
    print("len of train_labels:",len(train_labels))

    

    test    = zip(test_images,test_labels)
    train   = zip(train_images,train_labels)
    
    # train_images = torch.tensor(train['images'])
    # train_labels = torch.tensor(train['labels'])
    
    # test_images = torch.tensor(test['images'])
    # test_labels = torch.tensor(test['labels'])
    
    # trainloader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
    print(train_images[1])
    return train, test

_ = mnist()