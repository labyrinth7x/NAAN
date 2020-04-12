from feeder_test import Feeder_TEST
from gat import GAT
import argparse
import torch
from torch.utils.data import DataLoader
import logging
from utils import log_config, AverageMeter
import time
import os
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from utils import graph_propagation

logger = logging.getLogger()
logger.setLevel(logging.INFO)
eps = 1e-8
cuda = True




def accuracy(outputs, targets):
    preds = torch.argmax(outputs, dim=1).long()
    acc = torch.mean((preds == targets).float())
    tp = torch.sum((preds == targets).float() * targets.float()) + eps
    p = tp / (torch.sum((preds == 1).float())+eps)
    r = tp / (torch.sum((targets == 1).float())+eps)
    target1 = torch.sum((targets == 1)).float() / outputs.shape[0]
    return acc, p, r, target1



def main(args):
    knn_path = args.knn_path.replace('split1', args.split)
    feat_path = args.feat_path
    k_hops = args.k_hops
    N = np.load(knn_path).shape[0]
    split = args.split
    step = args.step
    sample_hops = args.sample_hops
    th = args.th
    print(split)
    logging.info(args)
    num_workers = args.num_workers

    model_path = args.model_path 
    checkpoint = torch.load(os.path.join(args.logs_dir, args.model_path))
    args = checkpoint['args']
    log_config(args, name='eval')
    #logging.info(args)

    
    args.logs_dir = os.path.join(args.logs_dir, model_path.split('.ckpt')[0]+'-hops-{}-{}-{}-step-{}-th-{}'.format(k_hops[0],k_hops[1],sample_hops,step, th))
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    temp_dir = args.logs_dir.split('-step-')[0] +'-'
    print(temp_dir+'{}_edges.npy'.format(split))
    if not os.path.exists(os.path.join(args.logs_dir, '{}_edges.npy'.format(split))) and not os.path.exists(temp_dir+'{}_edges.npy'.format(split)):
        net = GAT(args.embed_size, args.embed_size, args.dropout, args.nheads, args.alpha)
        classifier = nn.Sequential(
                    nn.Linear(1024, 512),
                    nn.PReLU(512),
                    nn.Linear(512,2))
        if cuda:
            net = net.cuda()
            classifier = classifier.cuda()

        net.load_state_dict({a.replace('module.',''):b for a,b in checkpoint['model'][0].items()})
        classifier.load_state_dict({a.replace('module.',''):b for a,b in checkpoint['model'][1].items()})
        criterion = nn.CrossEntropyLoss()
        if cuda:
            criterion = criterion.cuda()
        # Eval the Model
    
        net.eval()
        classifier.eval()
        
        testset = Feeder_TEST(feat_path, 
                      knn_path, 
                      k_hops, args.logs_dir, net, sample_hops, split)

    

        test_loader = DataLoader(testset, batch_size=204800, num_workers=num_workers, shuffle=False, pin_memory=True)


        edges, scores = validate(args, test_loader, [net, classifier], criterion)
        edges = np.array(edges)
        scores = np.array(scores)
        logging.info('computing edges&scores done')

        np.save(os.path.join(args.logs_dir, '{}_edges.npy'.format(split)), edges)
        np.save(os.path.join(args.logs_dir, '{}_scores.npy'.format(split)), scores)
    else:
        if os.path.exists(temp_dir+'{}_edges.npy'.format(split)):
            temp_dir = temp_dir+'{}_edges.npy'.format(split)
        else:
            temp_dir = os.path.join(args.logs_dir, '{}_edges.npy'.format(split))
        edges = np.load(temp_dir)
        scores = np.load(temp_dir.replace('edges','scores'))
        logging.info('Loading edges&scores done')


    logging.info('Start propagation')
   
    '''
    print(edges.shape)
    flag = scores > 0.5
    edges = edges[flag]
    scores = scores[flag]
    print(edges.shape)
    '''
    

    components = graph_propagation(edges, scores, th=th, max_sz = 600, step=step, max_iter=1000)

    # give examples "pesudo labels"
    cdp_res = []
    for c in components:
        cdp_res.append(sorted([n.name for n in c]))
    pred = -1 * np.ones(N, dtype=np.int)
    for i,c in enumerate(cdp_res):
        pred[np.array(c)] = i

    
    # rearrange according to the appearance order
    # pred = [4,1,4,1,0,2,3,-1,2] -> [0,1,0,1,2,3,4,-1,3]
    valid = np.where(pred != -1)
    _, unique_idx = np.unique(pred[valid], return_index=True)
    pred_unique = pred[valid][np.sort(unique_idx)]
    pred_mapping = dict(zip(list(pred_unique), range(pred_unique.shape[0])))
    pred_mapping[-1] = -1
    pred = np.array([pred_mapping[p] for p in pred])

    # analyse results
    num_valid = len(valid[0])
    num_class = len(pred_unique)
    logging.info('\n------------- Analysis --------------')
    logging.info('num_images: {}\tnum_class: {}\tnum_per_class: {:.2g}'.format(num_valid, num_class, num_valid/float(num_class)))
    logging.info("Discard ratio: {:.4g}".format(1 - num_valid / float(len(pred))))
    
    new_label = ['{}\n'.format(p) for p in pred]
    with open(os.path.join(args.logs_dir,'{}_labels.txt'.format(split)),'w') as f:
        f.writelines(new_label)
    
    np.save(os.path.join(args.logs_dir,'{}_labels.npy'.format(split)), pred)


def validate(args, test_loader, model, criterion):
    # average meters to record the training statistics

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs  = AverageMeter()
    precisions  = AverageMeter()
    recalls  = AverageMeter()  
    target1 = AverageMeter()
    
    net, classifier = model
    

    end = time.time()
    edges = []
    scores = []
    for i, test_data in tqdm(enumerate(test_loader)):
        # switch to train mode
        center_feature, onehop_feature, targets, center_idx, onehop_idx = test_data

        # measure data loading time
        data_time.update(time.time() - end)
        if cuda:
            center_feature = center_feature.cuda()
            onehop_feature = onehop_feature.cuda()
        
            targets = targets.cuda()

        # (batch_size, n_embeddings)
        #center = net([center_one, center_second])
        #onehop = net([one_one, one_second])

        features = torch.cat([center_feature,onehop_feature], dim=1)
        outputs = classifier(features)
        loss = criterion(outputs, targets)
        outputs = F.softmax(outputs,dim=1)

        acc, p, r, t1 = accuracy(outputs.cpu().detach(), targets.cpu().detach())

        losses.update(loss.item(), center_feature.shape[0])
        accs.update(acc, center_feature.shape[0])
        precisions.update(p, center_feature.shape[0])
        recalls.update(r, center_feature.shape[0])
        target1.update(t1, center_feature.shape[0])
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if i % args.log_step == 0:
            logging.info(
                'Epoch: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {losses.val:.3f} ({losses.avg:.3f})\t'
                'pos_num {target1.val:.3f} ({target1.val:.3f})\t'
                'Accuracy {accuracy.val:.3f} ({accuracy.avg:.3f})\t'
                'Precison {precisions.val:.3f} ({precisions.avg:.3f})\t'
                'Recall {recalls.val:.3f} ({recalls.avg:.3f})'
                .format(
                    i, len(test_loader), batch_time=batch_time,
                    losses=losses, target1=target1,
                    accuracy=accs, precisions=precisions, recalls=recalls))
        
        edges.extend(torch.stack([center_idx, onehop_idx], dim=1).numpy().tolist())
        scores.extend(outputs[:,1].cpu().detach().numpy().tolist())
        
    return edges, scores
        


def parse():

    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_path', default='data/labeled/split1_feats.npz',
                        help='path to feats')
    parser.add_argument('--knn_path', default='data/labeled/split1_knn.npy',
                        help='path to knn')
    parser.add_argument('--k_hops', type=int, nargs='+', default=[100,5])
    parser.add_argument('--num_epochs', default=5, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--embed_size', default=512, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--lr', default=.002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--wd', default=5e-4, type=float,
                        help='weight decay.')
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--alpha', default=0.2, type=float)
    parser.add_argument('--nheads', default=1, type=int)
    parser.add_argument('--lr_update', default=1, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--num_workers', default=1, type=int,
                        help='Number of 1ata loader workers.')
    parser.add_argument('--log_step', default=100, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--logs_dir', default='results_imgs',
                        help='Path to save Tensorboard log.')
    parser.add_argument('--model_path', help='Path to save Tensorboard log.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--split', type=str)
    parser.add_argument('--step', type=float,default=0.05)
    parser.add_argument('--sample_hops', type=int)
    parser.add_argument('--th', type=float,default=None)
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = parse()
    main(args)
