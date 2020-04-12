import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm
import logging
import time
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class Feeder(data.Dataset):
    def __init__(self, feat_path, knn_path, sample_hops, k_hop=[20,5]):
        fs = np.load(feat_path)
        self.features = fs['embs']
        self.labels = fs['labels']
        self.knn = np.load(knn_path)
        self.num_samples = len(self.features)
        self.k_hop = k_hop
        self.sample_hops = sample_hops

        pairs_path = os.path.join(os.path.dirname(knn_path), 'pairs_{}.npy'.format(sample_hops))
        print(pairs_path)
        if os.path.exists(pairs_path):
            self.pairs = np.load(pairs_path)
            logging.info('Loading {} pairs done'.format(len(self.pairs)))
        else:
            self.init(pairs_path)

    
    def init(self, pairs_path):
        logging.info('Loading all pairs.')
        self.pairs = []
        pairs = np.zeros((self.num_samples, self.num_samples),dtype=np.bool_)

        for index in tqdm(range(self.num_samples)):
            onehop_indices = self.knn[index][1:self.sample_hops+1]
            for idx in onehop_indices:
                if pairs[(index, idx)] == 1 or pairs[(idx, index)] == 1:
                    continue
                else:
                    self.pairs.append((index,idx))
                    pairs[(index, idx)] = 1
        logging.info('Loading {} pairs done'.format(len(self.pairs)))
        np.save(pairs_path, self.pairs)


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        

        center, onehop = self.pairs[index]

        target = (self.labels[center] == self.labels[onehop])

        # _onehop (201,512)
        # _secondhop (200,6,512)
        center_onehop, center_secondhop = self.get_features(center)
        onehop_onehop, onehop_secondhop = self.get_features(onehop)
            
        return center_onehop, center_secondhop, onehop_onehop, onehop_secondhop, int(target)


    def get_features(self, idx):
        onehop_indices = self.knn[idx][:self.k_hop[0]+1]
        
        secondhop_indices = []
        for onehop_idx in onehop_indices[1:]:
            secondhop_indices.append(self.knn[onehop_idx][:self.k_hop[1]+1])
        secondhop_indices = np.stack(secondhop_indices)

        onehop_features = self.features[onehop_indices]

        secondhop_indices = secondhop_indices.flatten()
        secondhop_features = self.features[secondhop_indices]
        secondhop_features = secondhop_features.reshape(self.k_hop[0],self.k_hop[1]+1,self.features.shape[1])
        
        return onehop_features, secondhop_features
