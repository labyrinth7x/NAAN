import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm
import logging
import time
import os
import redis
 
 
# r = redis.Redis(host='localhost', port=6379, db=0)
 
logger = logging.getLogger()
logger.setLevel(logging.INFO)
 
class Feeder_TEST(data.Dataset):
    def __init__(self, feat_path, knn_path, k_hop=[20,5], logs_dir=None, net=None, sample_hops=50, split='split1'):
         
        #fs = np.load(feat_path)
        #self.features = fs['embs']
        #self.labels = fs['labels']
        self.knn = np.load(knn_path)
        self.num_samples = len(self.knn)
        print(self.num_samples)
        self.k_hop = k_hop
        self.split = split
        self.sample_hops = sample_hops
 
        self.init_features(feat_path)
 
        #pairs_path = os.path.join(os.path.dirname(knn_path), 'pairs_{}.npy'.format(k_hop[0]))
        #pairs_path = os.path.join(logs_dir, '{}_pairs_{}_{}.npy'.format(self.split, k_hop[0], k_hop[1]))
        pairs_path = os.path.join(os.path.dirname(knn_path), '{}_pairs_{}.npy'.format(self.split, self.sample_hops))
        print(pairs_path)
        if os.path.exists(pairs_path):
            self.pairs = np.load(pairs_path)
            logging.info('Loading {} pairs done'.format(len(self.pairs)))
        else:
            self.init_pairs(pairs_path)
        # precompute all center features
        center_path = os.path.join(logs_dir, '{}_center_{}_{}.npy'.format(self.split, k_hop[0], k_hop[1]))
        print(center_path)
        if os.path.exists(center_path):
            self.centers = np.load(center_path)
            logging.info('Loading {} centers done'.format(len(self.centers)))
        else:
            self.init_centers(center_path, net)
 
    def init_features(self, feat_path):
        num_split = int(self.split[-1])
        if os.path.exists(feat_path.replace('split1', 'split1{}'.format(num_split))):
            info = np.load(feat_path)
            self.features = info['embs']
            self.labels = info['labels']
        self.features = []
        self.labels = []
        for i in tqdm(range(1, num_split+1)):
            info = np.load(feat_path.replace('split1', 'split{}'.format(i)))
            self.features.append(info['embs'])
            self.labels.append(info['labels'])
         
        self.features = np.concatenate(self.features)
        self.labels = np.concatenate(self.labels)
 
     
    def init_pairs(self, pairs_path):
        logging.info('Loading all pairs.')
        self.pairs = []
        #pairs = np.zeros((self.num_samples, self.num_samples),dtype=np.bool_)
 
        for index in tqdm(range(self.num_samples)):
            onehop_indices = self.knn[index][1:self.sample_hops+1]
            for idx in onehop_indices:
                temp = tuple(sorted([index,idx]))
                self.pairs.append(temp)
        self.pairs = list(set(self.pairs))
        logging.info('Loading {} pairs done'.format(len(self.pairs)))
        np.save(pairs_path, self.pairs)
 
    def init_centers(self, center_path, net):
        self.centers = np.zeros((self.num_samples, 512), dtype=np.float32)
        size = 512
        for i in tqdm(range(0, self.num_samples, size)):
            ass = []
            bss = []
            for j in range(i, min(i+size, self.num_samples)):
                a,b = self.get_features(j)
                ass.append(a)
                bss.append(b)
 
            ass = torch.Tensor(np.stack(ass)).cuda()
            bss = torch.Tensor(np.stack(bss)).cuda()
            center, _ = net([ass, bss])
            self.centers[i:min(i+size, self.num_samples)] = center.cpu().detach()
        logging.info('Loading {} centers done'.format(self.num_samples))
        print(center_path)
        np.save(center_path, self.centers)
 
 
    def __len__(self):
        return len(self.pairs)
 
    def __getitem__(self, index):
        center, onehop = self.pairs[index]
        center_feature = self.centers[center]
        onehop_feature = self.centers[onehop]
        target = (self.labels[center] == self.labels[onehop])
        #target = (int(np.frombuffer(r.get('label_{}'.format(center)))) == int(np.frombuffer(r.get('label_{}'.format(onehop)))))
        return center_feature, onehop_feature, int(target), int(center), int(onehop)
 
 
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
