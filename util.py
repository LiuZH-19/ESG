import pickle
import numpy as np
import pandas as pd 
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch.autograd import Variable
import json
import h5py

def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

def get_node_fea(data_set, train_num=0.6):
    if data_set == 'solar-energy':
        path = 'data/h5data/solar-energy.h5'
    elif data_set == 'electricity':
        path = 'data/h5data/electricity.h5'
    elif data_set == 'exchange-rate':
        path = 'data/h5data/exchange-rate.h5'
    elif data_set == 'wind':
        path = 'data/h5data/wind.h5'    
    elif data_set == 'nyc-bike':
        path = 'data/h5data/nyc-bike.h5'
    elif data_set == 'nyc-taxi':
        path = 'data/h5data/nyc-taxi.h5'
    else:
        raise ('No such dataset........................................')



    if data_set == 'nyc-bike' or data_set== 'nyc-taxi':
        x = h5py.File(path, 'r')
        data = list()
        for key in x.keys():
            data.append(x[key][:])
        data = np.stack(data, axis=1)
        num_train = 3001 #bike taxi
        df = data[:num_train]
        scaler = StandardScaler(df.mean(),df.std())
        train_feas = scaler.transform(df).reshape([-1,df.shape[2]])
    else:
        x = pd.read_hdf(path)
        data = x.values
        print(x.shape)
        num_samples = data.shape[0]
        num_train = round(num_samples * train_num)
        df = data[:num_train]
        print(df.shape)
        scaler = StandardScaler(df.mean(),df.std())
        train_feas = scaler.transform(df)
    return train_feas

class DataLoaderS(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, dataset, train, valid, device, horizon, window, normalize=2):
        self.P = window
        self.h = horizon
        filename = r'data/h5data/'+ dataset + '.h5'
        self.rawdat= (pd.read_hdf(filename)).to_numpy()
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        self.normalize = 2
        self.scale = np.ones(self.m)
        self._normalized(normalize)
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        self.scale = torch.from_numpy(self.scale).float()
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)

        self.scale = self.scale.to(device)
        self.scale = Variable(self.scale)

        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

        self.device = device

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if (normalize == 0):
            self.dat = self.rawdat

        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat)

        # normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))

    def _split(self, train, valid, test):

        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)

    def _batchify(self, idx_set, horizon):
        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, self.m))
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
            Y[i, :] = torch.from_numpy(self.dat[idx_set[i], :])
        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            X = X.to(self.device)
            Y = Y.to(self.device)
            yield Variable(X), Variable(Y)
            start_idx += batch_size

class DataLoaderM(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()
    
    def get_one(self, index):
        start_ind = self.batch_size * index
        end_ind = min(self.size, self.batch_size * (index + 1))
        x_i = self.xs[start_ind: end_ind, ...]
        y_i = self.ys[start_ind: end_ind, ...]
        return (x_i, y_i)


class StandardScaler():
    """
    Standard the input
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def transform(self, data):
        return (data - self.mean) / self.std
    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_dataset(dataset, batch_size, valid_batch_size= None, test_batch_size=None):
    data = {}
    dataset_dir = os.path.join('data', dataset)
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['raw_x_'+category] = cat_data['x']
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    if dataset == 'nyc-bike' or dataset =='nyc-taxi':
        #print('load_dataset : nyc'+"!"*30)
        scaler = StandardScaler(mean=data['x_train'].mean(), std=data['x_train'].std())
    else:
        scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        if dataset == 'nyc-bike' or dataset =='nyc-taxi':
            #print('load_dataset : nyc transform'+"!"*30)
            data['x_' + category] = scaler.transform(data['x_' + category])            
        else:
            data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
        print(category)
        print(data['x_' + category].shape)

    data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data



def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


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

def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs((preds-labels)/labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real, mask= False):
    assert pred.shape == real.shape , f'{pred.shape}, {real.shape}'
    mape = masked_mape(pred,real,0.0).item()
    if mask:
        mae = masked_mae(pred,real,0.0).item()    
        rmse = masked_rmse(pred,real,0.0).item()
    else:
        mae = masked_mae(pred,real).item()    
        rmse = masked_rmse(pred,real).item()
    return mae,mape,rmse


def load_node_feature(path):
    fi = open(path)
    x = []
    for li in fi:
        li = li.strip()
        li = li.split(",")
        e = [float(t) for t in li[1:]]
        x.append(e)
    x = np.array(x)
    mean = np.mean(x,axis=0)
    std = np.std(x,axis=0)
    z = torch.tensor((x-mean)/std,dtype=torch.float)
    return z


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)



            