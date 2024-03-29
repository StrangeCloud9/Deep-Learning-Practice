import paths
import config

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset

import csv
def writecsv(file,filename,flag=3):
    csvfile = open(filename, 'w')
    writer = csv.writer(csvfile,lineterminator='\n')
    #0 nodes , 1 = edgees
    for i, line in enumerate(file):
        writer.writerow((i,line))
    csvfile.close()
    
class myDataSet(Dataset):
    def __init__(self, x, y=None, padding_len=12):
        self.x = x
        self.y = y 
        self.is_test = False
        if y == None:
            self.is_test = True
        self.padding_len = padding_len
            
        self.len = 0

        for item in self.x:
            self.len += len(item)

        index = 0
        row = 0
        col = 0
        self.index_map= {}

        while index < self.len:
            while col >= len(self.x[row]):
                col = 0
                row += 1
            self.index_map[index] = [row, col]

            index += 1
            col += 1
        return 
    
    def __getitem__(self, real_idx):

        row = self.index_map[real_idx][0]
        index = self.index_map[real_idx][1]


        center_x = np.array(self.x[row][max(index - self.padding_len, 0) : min(index + self.padding_len + 1, len(self.x[row]))] )
        head_padding = max(0, 12 - index)
        tail_padding = max(0, 12 - (len(self.x[row]) - 1 - index))

        center_x = np.pad(center_x, ((head_padding, tail_padding), (0, 0)), 'constant', constant_values = 0)
        center_x = center_x.reshape(-1)
        if self.is_test: 
            return torch.tensor(center_x, dtype=torch.float)
        #torch.tensor(center_x, dtype=torch.float), torch.long
        return torch.tensor(center_x, dtype=torch.float), torch.tensor(self.y[row][index], dtype=torch.long)
        
    
    def __len__(self,):

        return self.len 


def get_loader(mode="train"):
    if mode == "train":
        data_path = paths.train_data
        labels_path = paths.train_labels
        shuffle = True
    if mode == "val":
        data_path = paths.valid_data
        labels_path = paths.valid_labels
        shuffle = False
    if mode == "test":
        data_path = paths.test_data
        labels_path = None
        shuffle = False
    data = np.load(data_path)

    if config.sanity:
        data = data[:100]

    if labels_path:
        labels = np.load(labels_path)
        if config.sanity:
            labels = labels[:100]

        print(data.shape, labels.shape)
        dataset = myDataSet(data, labels)
    else:
        dataset = myDataSet(data)

    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=config.batch_size, drop_last=True, pin_memory=True, num_workers=4)

    return dataloader


def get_test_labels():
    return np.load(paths.test_labels)
