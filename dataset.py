"""
Landmarks Dataset
    Class wrapper for interfacing with the dataset of landmark images
"""
import os
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.misc import imread, imresize
from torch.utils.data import Dataset, DataLoader
from utils import config
import string

def get_train_val_test_loaders(num_classes):
    tr, va, te, _ = get_train_val_dataset(num_classes=num_classes)
    
    batch_size = config('cnn.batch_size')
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False)
    
    return tr_loader, va_loader, te_loader, tr.get_semantic_label

def get_train_val_dataset(num_classes=10):
    tr = LandmarksDataset('train', num_classes)
    va = LandmarksDataset('val', num_classes)
    te = LandmarksDataset('test', num_classes)
    
    # Resize
    tr.X = resize(tr.X)
    va.X = resize(va.X)
    te.X = resize(te.X)
    
    # Standardize
    standardizer = ImageStandardizer()
    standardizer.fit(tr.X)
    tr.X = standardizer.transform(tr.X)
    va.X = standardizer.transform(va.X)
    te.X = standardizer.transform(te.X)
    
    # Transpose the dimensions from (N,H,W,C) to (N,C,H,W)
    tr.X = tr.X.transpose(0,3,1,2)
    va.X = va.X.transpose(0,3,1,2)
    te.X = te.X.transpose(0,3,1,2)
    
    return tr, va, te, standardizer

def resize(X):
    """
    Resizes the data partition X to the size specified in the config file.
    Uses bicubic interpolation for resizing.

    Returns:
        the resized images as a numpy array.
    """
    # TODO: Complete this function
    # X = np.array(X)
    image_dim = config('image_dim')
    resized = np.zeros((X.shape[0], image_dim, image_dim, 3))
    for i in range(X.shape[0]):
        resized[i] = imresize(X[i], size=(image_dim,image_dim), interp='bicubic')

    # print(resized.shape)
    return resized

class ImageStandardizer(object):
    """
    Channel-wise standardization for batch of images to mean 0 and variance 1. 
    The mean and standard deviation parameters are computed in `fit(X)` and
    applied using `transform(X)`.
    
    X has shape (N, image_height, image_width, color_channel)
    """
    def __init__(self):
        super().__init__()
        self.image_mean = None
        self.image_std = None
        self.dim = None
        self.N = None

    def extract_channel(self, X, flag):
        extract = np.zeros((self.N, self.dim, self.dim))
        for i in range(self.N):
            for j in range(self.dim):
                for k in range(self.dim):
                    extract[i][j][k] = X[i][j][k][flag]
        return extract

    def fit(self, X):
        # print(X.shape)
        self.N = X.shape[0]
        self.dim = X.shape[1]
        R = self.extract_channel(X, 0)
        # print(R.shape)
        G = self.extract_channel(X, 1)
        # print(G.shape)
        B = self.extract_channel(X, 2)
        # print(B.shape)

        self.image_mean = np.zeros(3)
        self.image_std = np.zeros(3)
        print("-----------extracting Red Layer-----------------")
        R = R.reshape((self.N, self.dim*self.dim))

        print("-----------extracting Green Layer-----------------")
        G = G.reshape((self.N, self.dim*self.dim))

        print("-----------extracting Blue Layer-----------------")
        B = B.reshape((self.N, self.dim*self.dim))

        self.image_mean[0] = np.mean(R)
        self.image_mean[1] = np.mean(G)
        self.image_mean[2] = np.mean(B)
        self.image_std[0] = np.std(R)
        self.image_std[1] = np.std(G)
        self.image_std[2] = np.std(B)

        print("------------------Problem 1 b(i)------------------------")
        print("The mean value for R is {0} and std is {1}".format(self.image_mean[0], self.image_std[0]))

        print("The mean value for G is {0} and std is {1}".format(self.image_mean[1], self.image_std[1]))

        print("The mean value for B is {0} and std is {1}".format(self.image_mean[2], self.image_std[2]))
        print("---------------------------------------------------------")
        print("------------------Problem 1 b(ii)------------------------")
        print("------------------handwritten------------------------")
        print("---------------------------------------------------------")

    def transform(self, X):
        print("------------------transforming data-----------------------")
        transformed = X
        for i in range(X.shape[0]):
            for j in range(self.dim):
                for k in range(self.dim):
                    for l in range(3):
                        transformed[i][j][k][l % 3] = (transformed[i][j][k][l % 3] - self.image_mean[l % 3])/self.image_std[l % 3]
        print("---------------------------------------------------------")
        return transformed


class LandmarksDataset(Dataset):

    def __init__(self, partition, num_classes=10):
        """
        Reads in the necessary data from disk.
        """
        super().__init__()
        
        if partition not in ['train', 'val', 'test']:
            raise ValueError('Partition {} does not exist'.format(partition))
        
        np.random.seed(0)
        self.partition = partition
        self.num_classes = num_classes
        
        # Load in all the data we need from disk
        self.metadata = pd.read_csv(config('csv_file'), index_col=0)
        self.X, self.y = self._load_data()
    
        self.semantic_labels = dict(zip(
            self.metadata['numeric_label'],
            self.metadata['semantic_label']
        ))

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), torch.tensor(self.y[idx]).long()
    
    def _load_data(self):
        """
        Loads a single data partition from file.
        """
        print("loading %s..." % self.partition)
        
        if self.partition == 'test':
            if self.num_classes == 5:
                df = self.metadata[self.metadata.partition == self.partition]
            elif self.num_classes == 10:
                df = self.metadata[self.metadata.partition.isin([self.partition, ' '])]
            else:
                raise ValueError('Unsupported test partition: num_classes must be 5 or 10')
        else:
            df = self.metadata[
                (self.metadata.numeric_label < self.num_classes) &
                (self.metadata.partition == self.partition)
            ]
        
        X, y = [], []
        for i, row in df.iterrows():
            label = row['numeric_label']
            image = imread(os.path.join(config('image_path'), row['filename']))
            X.append(image)
            y.append(row['numeric_label'])
        
        return np.array(X), np.array(y)

    def get_semantic_label(self, numeric_label):
        """
        Returns the string representation of the numeric class label (e.g.,
        the numberic label 1 maps to the semantic label 'miniature_poodle').
        """
        return self.semantic_labels[numeric_label]

if __name__ == '__main__':
    ## Future note: check scipy imread and imresize
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    np.set_printoptions(precision=3)
    tr, va, te, standardizer = get_train_val_dataset()
    print("Train:\t", len(tr.X))
    print("Val:\t", len(va.X))
    print("Test:\t", len(te.X))
    print("Mean:", standardizer.image_mean)
    print("Std: ", standardizer.image_std)
