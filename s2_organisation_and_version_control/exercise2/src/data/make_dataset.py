# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import numpy as np
import torch
from torch.nn.functional import normalize


# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def main():#input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    test = dict()
    train_images = []
    train_labels = []

    direct = "C:/Users/victo/OneDrive - Danmarks Tekniske Universitet/skole/9.semester/Machine Learning Operations 02476/dtu_mlops/s2_organisation_and_version_control/exercise2/data/raw/"
    direct2= "C:/Users/victo/OneDrive - Danmarks Tekniske Universitet/skole/9.semester/Machine Learning Operations 02476/dtu_mlops/s2_organisation_and_version_control/exercise2/data/processed/"
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

    train_images = torch.cat((train0_images, train1_images, train2_images, train3_images, train4_images))
    
    print("len of train_images:",len(train_images))

    t1 = normalize(train_images, p=1.0, dim = 1)
    torch.save(t1, f'{direct2}+tensor.pt')





if __name__ == '__main__':
    # log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # logging.basicConfig(level=logging.INFO, format=log_fmt)

    # # not used in this stub but often useful for finding various files
    # project_dir = Path(__file__).resolve().parents[2]

    # # find .env automagically by walking up directories until it's found, then
    # # load up the .env entries as environment variables

    main()
