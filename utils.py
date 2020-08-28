from datetime import datetime
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import io
from torchvision import transforms as trans
import torch
import pdb
import cv2
import logging
import os
from torch.utils.data.sampler import Sampler
 
#logger = logging.getLogger()
#logger.setLevel(logging.INFO)
 
class Data(object):
    def __init__(self, name):
        self.__name = name
        self.__links = set()
 
    @property
    def name(self):
        return self.__name
 
    @property
    def links(self):
        return set(self.__links)
 
    def add_link(self, other, score):
        self.__links.add(other)
        other.__links.add(self)
 
 
def get_time():
    return (str(datetime.now())[:-10]).replace(' ','-').replace(':','-')
 
class AverageMeter(object):
    """Computes and stores the average and current value"""
 
    def __init__(self):
        self.reset()
 
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
 
    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (1e-8 + self.count)
 
    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)
 
 
 
def log_config(args, **kwargs):
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    fpath = os.path.join(args.logs_dir, kwargs['name']+'.log')
    handler = logging.FileHandler(fpath)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(handler)
 
 
def connected_components_constraint(nodes, max_sz, score_dict=None, th=None):
    '''
    only use edges whose scores are above `th`
    if a component is larger than `max_sz`, all the nodes in this component are added into `remain` and returned for next iteration.
    '''
    result = []
    remain = set()
    nodes = set(nodes)
    while nodes:
        n = nodes.pop()
        group = {n}
        queue = [n]
        valid = True
        while queue:
            n = queue.pop(0)
            if th is not None:
                neighbors = {l for l in n.links if score_dict[tuple(sorted([n.name, l.name]))] >= th}
            else:
                neighbors = n.links
            neighbors.difference_update(group)
            nodes.difference_update(neighbors)
            group.update(neighbors)
            queue.extend(neighbors)
            if len(group) > max_sz or len(remain.intersection(neighbors)) > 0:
                # if this group is larger than `max_sz`, add the nodes into `remain`
                valid = False
                remain.update(group)
                break
        if valid: # if this group is smaller than or equal to `max_sz`, finalize it.
            result.append(group)
    #print("\tth: {}, remain: {}".format(th, len(remain)))
    return result, remain
 
def graph_propagation(edges, score, max_sz, th=None, step=0.1, max_iter=100):
 
    edges = np.sort(edges, axis=1)
    #th = score.min()
    #th = min(score)
    if th is -1:
        th = min(score)
         
 
    # construct graph
    score_dict = {} # score lookup table
    for i,e in enumerate(edges):
        score_dict[e[0], e[1]] = score[i]
 
    nodes = np.sort(np.unique(edges.flatten()))
    mapping = -1 * np.ones((nodes.max()+1), dtype=np.int)
    mapping[nodes] = np.arange(nodes.shape[0])
    link_idx = mapping[edges]
    vertex = [Data(n) for n in nodes]
    # firstly link all centers and their one-hop nodes.
    for l, s in zip(link_idx, score):
        vertex[l[0]].add_link(vertex[l[1]], s)
 
    # first iteration
    comps, remain = connected_components_constraint(vertex, max_sz)
 
    # iteration
    components = comps[:]
    Iter = 0
    logging.info('Done propagation initialization')
 
    while remain:
        th = th + (1 - th) * step
        comps, remain = connected_components_constraint(remain, max_sz, score_dict, th)
        components.extend(comps)
        print('Iter:{}'.format(Iter))
        Iter += 1
        if Iter >= max_iter:
            print("Warning: The iteration reaches max_iter: {}. Force stopped at: th {}, remain {} for efficiency. If you do not want it to be force stopped, please increase max_iter or set it to np.inf".format(max_iter, th, len(remain)))
            components.append(remain)
            remain = {}
    return components
