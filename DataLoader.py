import os
import numpy as np
from makedata import *
from maketable import *

np.random.seed(1513)

class DataLoader():
    
    def __init__ (self, dir_path, subdivision):
        # Process the panoramic picture
        self.dataset = []
        cnt = 0;
        for file in sorted(os.listdir(dir_path)):
            if file[-4:] == '.png' or file[-4:] == '.jpg':
                # subdivision defualt = 8
                self.dataset.append(pano2icosa(os.path.join(dir_path, file), subdivision))
                print('Process %05d pictures..\r' %cnt, end='')
                cnt += 1

        self.dataset = np.array(self.dataset) # shape = (N, 4**8 * 20, 3)
        self.num_data = self.dataset.shape[0]

    def shuffle(self):
        self.rng = np.random.shuffle(np.arange(self.num_data))
        self.dataset = self.dataset[self.rng]


def main():
    pano_data = DataLoader('/media/bl530/新增磁碟區/area_1/pano/rgb/', 8)
    np.save('./area_1.npy', pano_data.dataset)

if __name__ == '__main__':
    main()