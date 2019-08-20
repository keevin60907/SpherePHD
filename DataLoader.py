import os
import numpy as np
import cv2
from utils import *
from makedata import *
from maketable import *

CLASS_LIST = ['<UNK>', 'beam', 'board', 'bookcase', 'ceiling', 'chair', 'clutter',
              'column', 'door', 'floor', 'sofa', 'table', 'wall', 'window']
ID_LIST = [1, 160, 297, 881, 1266, 2629, 6511, 6765, 7308, 7592, 7647, 8102, 9648, 9816]
np.random.seed(1513)

class DataLoader():
    
    def __init__ (self, dir_path, subdivision):
        # Process the panoramic picture
        self.dataset = []
        cnt = 1
        for file in sorted(os.listdir(dir_path)):
            if file[-4:] == '.png' or file[-4:] == '.jpg':
                # subdivision defualt = 8
                img = cv2.imread(os.path.join(dir_path, file))
                img = cv2.resize(img, (400, 200))
                self.dataset.append(pano2icosa(img, subdivision))
                print('Process %05d pictures..\r' %cnt, end='')
                cnt += 1

        self.dataset = np.array(self.dataset) # shape = (N, 4**8 * 20, 3)
        self.dataset = self.dataset / 255
        self.num_data = self.dataset.shape[0]

    def shuffle(self):
        self.rng = np.random.shuffle(np.arange(self.num_data))
        self.dataset = self.dataset[self.rng]

    def make_label(self, subdivision):
        reference = load_labels('./assets/semantic_labels.json')
        cnt = 1
        ret = []
        for file in sorted(os.listdir('/media/bl530/新增磁碟區/area_1/pano/semantic/')):
            if file[-4:] == '.png' or file[-4:] == '.jpg':
                img = cv2.imread(os.path.join('/media/bl530/新增磁碟區/area_1/pano/semantic/', file))
                img = cv2.resize(img, (400, 200))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                tmp = np.zeros((img.shape[0], img.shape[1], 1))-1
                print('Label %05d pictures..\r' %cnt, end='')
                index = img[:, :, 0]*256*256 + img[:, :, 1]*256 + img[:, :, 2]
                index = np.expand_dims(index, axis=2)
                index[index > 9816] = 0
                for i in range(len(CLASS_LIST)):
                    tmp[(index < ID_LIST[i]) & (tmp == -1)] = i
                ret.append(pano2icosa(tmp, subdivision))
                cnt += 1
        print('')
        return np.array(ret)

def main():
    pano_data_1 = DataLoader('/media/bl530/新增磁碟區/area_1/pano/rgb/', 6)
    print('')
    pano_data_2 = DataLoader('/media/bl530/新增磁碟區/area_2/pano/rgb/', 6)
    print('')
    pano_data_3 = DataLoader('/media/bl530/新增磁碟區/area_3/pano/rgb/', 6)
    print('')
    pano_data_4 = DataLoader('/media/bl530/新增磁碟區/area_4/pano/rgb/', 6)
    print('')
    pano_data_6 = DataLoader('/media/bl530/新增磁碟區/area_6/pano/rgb/', 6)
    print('')
    train_data = np.concatenate([pano_data_1.dataset, pano_data_2.dataset, pano_data_3.dataset,\
                                 pano_data_4.dataset, pano_data_6.dataset], axis=0)
    np.save('./train_data.npy', train_data)
    print('')

    label_1 = pano_data_1.make_label(6)
    print('')
    label_2 = pano_data_2.make_label(6)
    print('')
    label_3 = pano_data_3.make_label(6)
    print('')
    label_4 = pano_data_4.make_label(6)
    print('')
    label_6 = pano_data_6.make_label(6)
    print('')
    train_label = np.concatenate([label_1, label_2, label_3, label_4, label_6], axis=0)
    print('')
    np.save('./train_label.npy', train_label)
    print('')

    pano_data_5a = DataLoader('/media/bl530/新增磁碟區/area_5a/pano/rgb/', 6)
    print('')
    pano_data_5b = DataLoader('/media/bl530/新增磁碟區/area_5b/pano/rgb/', 6)
    print('')
    train_data = np.concatenate([pano_data_5a.dataset, pano_data_5b.dataset], axis=0)
    np.save('./test_data.npy', test_data)
    print('')

    label_5a = pano_data_5a.make_label(6)
    print('')
    label_5b = pano_data_5b.make_label(6)
    print('')
    train_label = np.concatenate([label_5a, label_5b], axis=0)
    print('')
    np.save('./test_label.npy', test_label)
    print('')

if __name__ == '__main__':
    main()
