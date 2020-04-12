import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init



class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, alpha):
        super(GATLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        self.w = nn.Linear(in_dim, out_dim)
        self.relu = nn.LeakyReLU(alpha)
        self.a = nn.Linear(out_dim * 2, 1)
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        self.init_weights()

    def init_weights(self):
        init.xavier_uniform_(self.w.weight)
        init.xavier_uniform_(self.a.weight)
        init.constant_(self.a.bias, 0)
        init.constant_(self.w.bias, 0)
        init.constant_(self.bias, 0)


    def forward(self, features):

        # features (batch_size, k+1, n_embeddings)
        
        h = self.w(features)

        h_expand = h[:,0:1,:]
        h_expand = h_expand.repeat(1,h.shape[1],1)
        
        e = self.relu(self.a(torch.cat([h_expand, h], dim=2)))

        attention = nn.Softmax(dim=1)(e)

        attention = self.dropout(attention)

        # h (batch_size, 1, embeddings)
        h = torch.matmul(attention.transpose(1,2), h)
        h = F.relu(h+self.bias)
        
        return h



class GAT(nn.Module):
    def __init__(self, in_dim=512, out_dim=512, dropout=0, nheads=4, alpha=0.2):
        super(GAT, self).__init__()


        attention = [GATLayer(in_dim, out_dim, dropout, alpha) for _ in range(nheads)]
        self.attention = nn.Sequential()

        for idx, att in enumerate(attention):
            self.attention.add_module('attn_{}'.format(idx), att)


    def forward(self, inputs):
        # onehop_features (batch_size, k1+1, n_embeddings)
        # secondhop_features (batch_size, k1, k2+1, n_embeddings)
        onehop_features, secondhop_features = inputs

        # first steps
        '''
        center_featuress = torch.stack([attn(onehop_features) for attn in self.attention], dim=1)
        # (batch_size, 1, embeddings)
        center_featuress = torch.mean(center_featuress, dim=1)
        '''
        # secondhop_featueres -> (batch_size * k1, k2+1, n_embeddings)
        secondhop_featuress = secondhop_features.reshape(-1, secondhop_features.shape[2], secondhop_features.shape[3])
        # (batch_size*k1, 1, n_embeddings) -> (batch_size*k1, nheads, n_embeddings)
        secondhop_featuress = torch.stack([attn(secondhop_featuress) for attn in self.attention], dim=1)
        secondhop_featuress = torch.mean(secondhop_featuress, dim=1)
        # (batch_size, k1,  n_embeddings)
        onehop_featuress = secondhop_featuress.view(onehop_features.shape[0], -1, secondhop_featuress.shape[2])

        # second steps
        # (batch_siz, k1+1, n_embeddings)
        #center_features = torch.cat([center_featuress, onehop_featuress], dim=1)
        center_features = torch.cat([onehop_features[:,0:1,:], onehop_featuress], dim=1)

        # (batch_size, 1, n_embeddings)
        center_features = torch.stack([attn(center_features) for attn in self.attention], dim=1)
        center_features = torch.mean(center_features, dim=1).squeeze(1)

        return center_features
