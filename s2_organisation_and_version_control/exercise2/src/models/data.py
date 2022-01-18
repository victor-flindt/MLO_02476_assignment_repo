import glob
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def mnist():

    class Train_data(Dataset):
        def __init__(self):
            self.imgs_path = "train/"
            file_list = glob.glob(self.imgs_path + "*")
            self.data = []
            for class_path in file_list:
                class_name = class_path.split("\\")[-1]
#                print(class_path)
                for img_path in glob.glob(class_path + "\\*.jpg"):
                    self.data.append([img_path, class_name])
            self.class_map = {"0" : 0, "1": 1,"2": 2,"3": 3,"4": 4,"5": 5,"6": 6,"7": 7,"8": 8,"9": 9}

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            img_path, class_name = self.data[idx]
            img = Image.open(img_path).convert('L')
            
            img = np.array(img)/255

            class_id = self.class_map[class_name]
            img_tensor = torch.from_numpy(img)
            img_tensor = img_tensor.float()
            img_tensor = torch.unsqueeze(img_tensor, 0)

            #img_tensor = torch.flatten(img_tensor)
            #img_tensor = img_tensor.permute(2, 1, 0)
            #class_id = torch.tensor([class_id])
            return img_tensor, class_id

    train_data = Train_data()
    #test_data = Test_data()
    #trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    return train_data#,test_data


_ = mnist()
