import sys
import numpy as np
import os
from PIL import Image
import torch
from torchvision import transforms as trans
from scipy.sparse import coo_matrix
from tqdm import tqdm
import logging
import argparse

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def remove_single(info):
    info = {a:b for a,b in info.items() if b != 1 and a != -1}
    return info



def print_generated(dir):
    # print #classes after removing singles
    lines = open(dir,'r').readlines()
    info = {}
    for line in lines:
        if int(line) in info:
            info[int(line)] += 1
        else:
            info[int(line)] = 1
    if '-1' in info:
        print('generated class:{}'.format(len(info)-1))
    else:
        print('generated class:{}'.format(len(info)))
    info = remove_single(info)
    print('after removing single:{}'.format(len(info)))


def list2dict(l):
    result = {}
    for a in l:
        a = int(a)
        if a in result:
            result[a] += 1
        else:
            result[a] = 1
    return result




def evaluate(generated, true, remove=True):
    # print pseudo ac
    tp = 0
    lines1 = open(true, 'r').readlines()
    t = [0]
    for idx in range(1, len(lines1)):
        if lines1[idx] == lines1[idx-1]:
            if idx == len(lines1) - 1:
                t.append(idx)
            continue
        t.append(idx)
    print('true classes:{}'.format(len(t)-1))

    lines2 = open(generated, 'r').readlines()
    dic = list2dict(lines2)
    if remove:
        dic = remove_single(dic)

    for idx in range(1, len(t)):
        rangee = lines2[t[idx-1]:t[idx]]
        rangee = list2dict(rangee)
        times, cls = max(zip(rangee.values(),rangee.keys()))
        if cls not in dic.keys():
            continue
        tp += times
    print('predict:{:.4f} ->> {}/{}'.format(tp / sum(dic.values()), tp, sum(dic.values())))
    print('discard ratio:{}'.format(1-sum(dic.values()) / len(lines1)))
    


def class2idx(classes, dic):
    result = {}
    flag = 0
    for c in classes:
        if c not in dic:
            continue
        if c in result:
            continue
        result[c] = flag
        flag += 1
    return result


def remove(dir):
    lines = open(dir, 'r').readlines()
    lines = [int(item.split('\n')[0]) for item in lines]
    dic = list2dict(lines)
    dic = remove_single(dic)
    c2i = class2idx(lines, dic)
    result = []
    for idx, line in enumerate(lines):
        if line in c2i:
            result.append(idx)
    return result
   

def removes(dir, split):
    lines = open(dir, 'r').readlines()
    lines = [int(item.split('\n')[0]) for item in lines]
    dic = list2dict(lines)
    dic = remove_single(dic)
    c2i = class2idx(lines, dic)
    result = []
    flag = 0
    for line in lines:
        if line not in c2i:
            result.append('-1'+ '\n')
        else:
            flag += 1
            result.append(str(c2i[line]) + '\n')
   
    dst = os.path.join(os.path.dirname(dir),'{}_labels_clean.txt'.format(split))
    #print(dst)
    f = open(dst, 'w')
    f.writelines(result)


def contingency_matrix(ref_labels, sys_labels):
    """Return contingency matrix between ``ref_labels`` and ``sys_labels``."""
    ref_classes, ref_class_inds = np.unique(ref_labels, return_inverse=True)
    sys_classes, sys_class_inds = np.unique(sys_labels, return_inverse=True)
    n_frames = ref_labels.size
    # Following works because coo_matrix sums duplicate entries. Is roughly
    # twice as fast as np.histogram2d.
    cmatrix = coo_matrix(
        (np.ones(n_frames), (ref_class_inds, sys_class_inds)),
        shape=(ref_classes.size, sys_classes.size),
        dtype=np.int)
    cmatrix = cmatrix.toarray()
    return cmatrix, ref_classes, sys_classes


def bcubed(ref_labels, sys_labels, cm=None):
    """Return B-cubed precision, recall, and F1.
    The B-cubed precision of an item is the proportion of items with its
    system label that share its reference label (Bagga and Baldwin, 1998).
    Similarly, the B-cubed recall of an item is the proportion of items
    with its reference label that share its system label. The overall B-cubed
    precision and recall, then, are the means of the precision and recall for
    each item.
    Parameters
    ----------
    ref_labels : ndarray, (n_frames,)
        Reference labels.
    sys_labels : ndarray, (n_frames,)
        System labels.
    cm : ndarray, (n_ref_classes, n_sys_classes)
        Contingency matrix between reference and system labelings. If None,
        will be computed automatically from ``ref_labels`` and ``sys_labels``.
        Otherwise, the given value will be used and ``ref_labels`` and
        ``sys_labels`` ignored.
        (Default: None)
    Returns
    -------
    precision : float
        B-cubed precision.
    recall : float
        B-cubed recall.
    f1 : float
        B-cubed F1.
    References
    ----------
    Bagga, A. and Baldwin, B. (1998). "Algorithms for scoring coreference
    chains." Proceedings of LREC 1998.
    """
    if cm is None:
        cm, _, _ = contingency_matrix(ref_labels, sys_labels)
    cm = cm.astype('float64')
    cm_norm = cm / cm.sum()
    precision = np.sum(cm_norm * (cm / cm.sum(axis=0)))
    recall = np.sum(cm_norm * (cm / np.expand_dims(cm.sum(axis=1), 1)))
    f1 = 2*(precision*recall)/(precision + recall)
    return precision, recall, f1



    
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('pseudo_dir', default='pseudo_labels',
                        help='dirname of the generated pseudo labels')
    parser.add_argument('split', default='split1',
                        help='the testing split')
    args = parser.parse_args()

    base_file = args.pseudo_dir + '/{}_labels.txt'.format(args.split)
    print_generated(base)
    gt_file = 'data/unlabeled/{}_labels.txt'.format(args.split)
    evaluate(base_file, gt_file, True)
    removes(base_file, args.split)

    base = open(base_file,'r').readlines()
    base_ori = np.array([int(item) for item in base])
    gt = open(com,'r').readlines()
    gt_ori = np.array([int(item) for item in gt])
    print('start evaluation')
    p,r,f = bcubed(gt_ori, base_ori)
    print('{:.4f},{:.4f},{:.4f}'.format(p,r,f))
    # remove single
    index = remove(base_file)
    base_rmv = base_ori[index]
    gt_rmv = gt_ori[index]
    print('start evaluation with singleton removing')
    p_rmv, r_rmv, f_rmv = bcubed(gt_rmv, base_rmv)
    print('{:.4f},{:.4f},{:.4f}'.format(p_rmv, r_rmv, f_rmv))
