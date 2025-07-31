import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

# metric
def metric(pred, label):
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        wape = np.divide(np.sum(mae), np.sum(label))
        wape = np.nan_to_num(wape * mask)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return mae, rmse, mape

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def _compute_loss(y_true, y_predicted):
        return masked_mae(y_predicted, y_true, 0.0)

def seq2instance(data, P, Q):
    num_step, nodes, dims = data.shape
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape = (num_sample, P, nodes, dims))
    y = np.zeros(shape = (num_sample, P, nodes, dims))
    for i in range(num_sample):
        x[i] = data[i : i + P]
        y[i] = data[i + Q : i + P + Q]
    return x, y

def read_meta(path):
    meta = pd.read_csv(path)
    lat = meta['Lat'].values
    lng = meta['Lng'].values
    locations = np.stack([lat,lng], 0)
    return locations

def construct_adj(data, num_node):
    # construct the adj through the cosine similarity
    data_mean = np.mean([data[24*12*i: 24*12*(i+1)] for i in range(data.shape[0]//(24*12))], axis=0)
    data_mean = data_mean.squeeze().T
    tem_matrix = cosine_similarity(data_mean, data_mean)
    tem_matrix = np.exp((tem_matrix-tem_matrix.mean())/tem_matrix.std())
    return tem_matrix


def loadData(filepath, metapath, P, Q, train_ratio, test_ratio, adjpath, tod, dow, log):
    # Traffic
    Traffic = np.load(filepath)['data'][..., :1]
    locations = read_meta(metapath)
    num_step = Traffic.shape[0]
    
    # temporal positions
    TE = np.zeros([num_step, 2])
    TE[:, 0] = np.array([i % tod for i in range(num_step)])
    TE[:, 1] = np.array([(i // tod) % dow for i in range(num_step)])
    TE_tile = np.repeat(np.expand_dims(TE, 1), Traffic.shape[1], 1)

    log_string(log, f'Shape of data: {Traffic.shape}')
    log_string(log, f'Shape of locations: {locations.shape}')
    log_string(log, f'Shape of TE: {TE_tile.shape}')

    # train/val/test 
    train_steps = round(train_ratio * num_step)
    test_steps = round(test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps
    trainData, trainTE = Traffic[:train_steps], TE_tile[:train_steps]
    valData, valTE = Traffic[train_steps: train_steps + val_steps], TE_tile[train_steps: train_steps + val_steps]
    testData, testTE = Traffic[-test_steps:], TE_tile[-test_steps:]

    # load adj for padding
    if os.path.exists(adjpath):
        adj = np.load(adjpath)
    else:
        adj = construct_adj(trainData, locations.shape[1])
        np.save(adjpath, adj)
    
    train_dataset = TrafficDataset(trainData, trainTE, P, Q)
    val_dataset = TrafficDataset(valData, valTE, P, Q)
    test_dataset = TrafficDataset(testData, testTE, P, Q)


    # normalization
    mean, std = np.mean(trainData), np.std(trainData)
    tmean, tstd = np.mean(testData), np.std(testData)

    lat = locations[0]  # shape: [n]
    lng = locations[1]  # shape: [n]

    delta_lat = -(lat[None, :] - lat[:, None]) + 0.025  # shape: [n, n]
    delta_lng = (lng[None, :] - lng[:, None]) + 0.025   # shape: [n, n]

    position_relat = np.stack([delta_lat, delta_lng], axis=-1)


    # log
    log_string(log, f'Shape of Train: {len(train_dataset)}')
    log_string(log, f'Shape of Validation: {len(val_dataset)}')
    log_string(log, f'Shape of Test: {len(test_dataset)}')
    log_string(log, f'Train Mean: {mean} & Std: {std}')
    log_string(log, f'Test Mean: {tmean} & Std: {tstd}')

    return train_dataset, val_dataset, test_dataset,  mean, std, tmean, tstd, position_relat, locations


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
import os

class ImageFolderDataset(Dataset):
    def __init__(self, img_dir, img_size):
        self.img_dir = img_dir
        self.img_files = sorted(
            [f for f in os.listdir(img_dir) if f.endswith('.png')],
            key=lambda x: int(os.path.splitext(x)[0])
        )
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ConvertImageDtype(torch.float32),  # [0, 255] -> [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = read_image(img_path) 
        if img.shape[0] == 4:        
            img = img[:3]             
        img = self.transform(img)  
        return img

import torch
from torch.utils.data import Dataset

class TrafficDataset(Dataset):
    def __init__(self, data, TE, P, Q):
        self.data = data
        self.TE = TE
        self.P = P
        self.Q = Q
        self.length = data.shape[0] - P - Q + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.P]
        y = self.data[idx + self.Q:idx + self.P + self.Q]
        x_te = self.TE[idx:idx + self.P]
        y_te = self.TE[idx + self.Q:idx + self.P + self.Q]
        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(y).float(),
            torch.from_numpy(x_te).float(),
            torch.from_numpy(y_te).float()
        )
