import torch
import numpy as np
from torch.utils.data import Dataset


class WindPMFDataset(Dataset):
    def __init__(self, X, Y):
        super(WindPMFDataset, self).__init__()
        self.X = X 
        self.Y = Y 
    
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        data = torch.tensor(self.X[index, ::]).float()
        labels = torch.tensor(self.Y[index, ::]).float()
        return data, labels

def parse(file):
    with open(f'tcnn_data/{file}') as f:
        x = []
        for i in range(12):
            x.append(f.readline().split())

    x = np.array(x)
    x = x.astype('float32')
    return x

def load_data(
        batch_size, val_batch_size,
        data_root, num_workers):

    target = []
    for file in os.listdir('tcnn_data'):
        target.append(parse(file))
    target = np.array(target)

    # prepare input and target arrays
    image_1d = target.reshape(-1,192)
    X = []
    y = []
    for i in range(target.shape[0]-8):
        X.append(image_1d[i:i+8,:])
        y.append(target[i+8,:])

    X = np.array(X)
    # X = X.reshape(X.shape[0], 8, target.shape[1], target.shape[2], 1)
    y = np.array(y)
    # y = y.reshape(X.shape[0], target.shape[1], target.shape[2], 1)

    train_size = int(X.shape[0]*0.8)
    X_train, Y_train = X[:train_size], y[:train_size]
    X_test, Y_test = X[train_size:], y[train_size:]

    train_set = WindPMFDataset(X=X_train, Y=Y_train)
    test_set = WindPMFDataset(X=X_test, Y=Y_test)

    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    return dataloader_train, None, dataloader_test, 0, 1