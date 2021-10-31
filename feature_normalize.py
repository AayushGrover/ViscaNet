import numpy as np

from glob import glob
from tqdm import tqdm

from chemprop.features import load_features
from chemprop.data.scaler import StandardScaler

def get_dist(dirpath):
    mean = list()
    std = list()
    count = list()

    # Get mean and standard deviation of all the features across all the files
    for fname in tqdm(sorted(glob(dirpath))):
        feats = load_features(fname)
        X = np.array(feats).astype(float)
        means = np.nanmean(X, axis=0)
        stds = np.nanstd(X, axis=0)
        means = np.where(np.isnan(means), np.zeros(means.shape), means)
        stds = np.where(np.isnan(stds), np.ones(stds.shape), stds)
        stds = np.where(stds == 0, np.ones(stds.shape), stds)
        mean.append(means)
        std.append(stds)
        count.append(X.shape[0])
    
    return mean, std, count

def get_overall_dist(mean, std, count):
    
    total = sum(count)
    total_mean = np.zeros(mean[0].shape)
    
    # Compute overall mean
    for i in range(len(count)):
        total_mean += (count[i]*mean[i])
    total_mean = total_mean/total

    term1 = np.zeros(std[0].shape)
    term2 = np.zeros(std[0].shape)

    # Compute overall standard deviation
    for i in range(len(count)):
        term1 += (count[i]*(std[i]**2))
        term2 += (count[i]*((mean[i]-total_mean)**2)) 

    total_std = ((term1 + term2)/total)**(0.5)
    return total_mean, total_std

def normalize(dirpath, savepath, mean, std):
    sc = StandardScaler(means=mean, stds=std, replace_nan_token=0)
    for _,fname in tqdm(enumerate(sorted(glob(dirpath)))):
        name = fname.split('/')[-1]
        # if 'fda' in name:
        feats = load_features(fname)
        feats = sc.transform(feats)
        np.save(savepath+name,feats)

if __name__ == "__main__":
    mean, std, count = get_dist('./data/new_data_feats/train/df*.npy')
    mean, std = get_overall_dist(mean, std, count)
    normalize('./data/fda.npy', './data/new_data_norm_feats/test/', mean, std)
    # normalize('./data/new_data_feats/*.npy', './data/new_data_norm_feats/', mean, std)
