"""
LFW dataloading
"""
import argparse
import time
import numpy as np
import torch
import glob
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os, os.path
import matplotlib.pyplot as plt

class LFWDataset(Dataset):
    def __init__(self, path_to_folder: str, transform) -> None:
        # TODO: fill out with what you need
        self.path = path_to_folder
        self.samples =  glob.glob(self.path+"\\*\\*.jpg")
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> torch.Tensor:
        # TODO: fill out
        
        img = Image.open(self.samples[index])

        return self.transform(img)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_to_folder', default='data', type=str)
    parser.add_argument('-num_workers', default=1, type=int)
    parser.add_argument('-visualize_batch', action='store_true')
    parser.add_argument('-get_timing', action='store_true')
    args = parser.parse_args()
    
    lfw_trans = transforms.Compose([
                transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
                transforms.ToTensor()
    ])
    
    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)
    
    # Define dataloader
    # Note we need a high batch size to see an effect of using many
    # number of workers
    BATCH_SIZE = 512
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=args.num_workers)

    if args.visualize_batch:
        fig, axs = plt.subplots(5)
        for batch_idx, inputs in enumerate(next(iter(dataloader))):
            if batch_idx >= 5:
                break
            else:
                print(batch_idx)
                axs[batch_idx].imshow(np.transpose(inputs.numpy(),(1,2,0)))
        plt.show()

    if args.get_timing:
        # lets do so repetitions
        res = []
        print("inde")
        for _ in range(1):
            start = time.time()
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx > 100:
                    break
            end = time.time()

            res.append(end - start)
            
        res = np.array(res)
        print(f'Timing: {np.mean(res)}+-{np.std(res)}')
