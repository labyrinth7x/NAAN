import argparse
import torch
from torch.utils.data import DataLoader
import logging
from utils import log_config, AverageMeter
import time
import os
import torch.nn as nn
from data.data_pipe import SubsetSampler
import numpy as np
from torch.nn.utils.clip_grad import clip_grad_norm_
from feeder1 import Feeder
from gat import GAT


logger = logging.getLogger()
logger.setLevel(logging.INFO)
eps = 1e-8


def save_checkpoint(state, fpath):
    tries = 15
    error = None

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            torch.save(state, fpath)
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('model save {} failed, remaining {} trials'.format(fpath, tries))
        if not tries:
            raise error

def accuracy(outputs, targets):
    preds = torch.argmax(outputs, dim=1).long()
    acc = torch.mean((preds == targets).float())
    tp = torch.sum((preds == targets).float() * targets.float()) + eps
    p = tp / (torch.sum((preds == 1).float())+eps)
    r = tp / (torch.sum((targets == 1).float())+eps)
    ratio = torch.sum((targets == 1).float()) / len(targets)
    return acc, p, r, ratio


def adjust_lr(args, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    

def main(args):

    log_config(args, name='train')
    logging.info(args)
    
    
    device = list(range(torch.cuda.device_count()))

    net = GAT(args.embed_size, args.embed_size, args.dropout, args.nheads, args.alpha)

    
    classifier = nn.Sequential(
                    nn.Linear(1024, 512),
                    nn.PReLU(512),
                    nn.Linear(512,2))
    
    if len(device) > 1:
        net = nn.DataParallel(net, device_ids=device).cuda()
        classifier = nn.DataParallel(classifier, device_ids=device).cuda()
    else:    
        net = net.cuda()
        classifier = classifier.cuda()


    start_step = 0

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            net.load_state_dict(checkpoint['model'][0])
            classifier.load_state_dict(checkpoint['model'][1])
            if 'step' in args.resume:
                start_epoch = checkpoint['epoch']
                start_step = checkpoint['step'] + 1
            else:
                start_epoch = checkpoint['epoch'] + 1
            
            args = checkpoint['args']
            # Eiters is used to show logs as the continuation of another
            # training
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
        
        trainset = Feeder(args.feat_path, 
                      args.knn_path, 
                      args.sample_hops, args.k_hops)
        train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)
    
    else:
        save_checkpoint({
            'epoch': 0,
            'model':[net.state_dict(), classifier.state_dict()],
            'args':args,
            }, fpath=os.path.join(args.logs_dir, 'epoch_{}.ckpt'.format(0)))
        start_epoch = 0
        start_step = 0
        if 'new1' in args.logs_dir:
            from feeder1 import Feeder
            trainset = Feeder(args.feat_path, 
                      args.knn_path, 
                      args.sample_hops, args.k_hops)
        else:
            from feeder import Feeder
            trainset = Feeder(args.feat_path, 
                      args.knn_path, 
                      args.k_hops)
        
        train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)

    
    N = len(train_loader)
    
    if start_step == 0:
        iters = list(train_loader.batch_sampler)
        #save_index
        np.save(os.path.join(args.logs_dir, 'epoch_{}_iters.npy'.format(start_epoch)) ,iters)
    else:
        iters = np.load(os.path.join(args.logs_dir, 'epoch_{}_iters.npy'.format(start_epoch)), allow_pickle=True)[start_step:]
        print('loading iters done')
        train_loader = DataLoader(trainset, batch_sampler=SubsetSampler(iters), num_workers=args.workers, pin_memory=True)

    params = list(net.parameters())
    params += list(classifier.parameters())
    optimizer = torch.optim.Adam(params, lr = args.lr, weight_decay = args.wd)
    criterion = nn.CrossEntropyLoss().cuda()


    # Train the Model
    for epoch in range(start_epoch, args.num_epochs):

        
        adjust_lr(args, optimizer, epoch)

        # train for one epoch
        train(args, train_loader, [net, classifier], (epoch, start_step, N), optimizer, criterion, params)


        save_checkpoint({
            'epoch': epoch,
            'model': [net.state_dict(), classifier.state_dict()],
            'args': args,
        }, fpath=os.path.join(args.logs_dir, 'epoch_{}.ckpt'.format(epoch)))
        
        if start_step != 0:
            start_step = 0
            train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)

        np.save(os.path.join(args.logs_dir, 'epoch_{}_iters.npy'.format(epoch+1)) ,iters)


def train(args, train_loader, model, info, optimizer, criterion, params):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()
    ratios = AverageMeter()

    net, classifier = model
    net.train()
    classifier.train()

    epoch, step, N = info
    

    end = time.time()
    for i, train_data in enumerate(train_loader):
        # switch to train mode
        # (batch_size, k_hop[0]+1, 512)
        # (batch_size, k_hop[0], k_hop[1]+1, 512)
        center_one, center_second, one_one, one_second, targets = train_data

        # measure data loading time
        data_time.update(time.time() - end)
        
        center_one = center_one.cuda()
        center_second = center_second.cuda()
        one_one = one_one.cuda()
        one_second = one_second.cuda()
        targets = targets.cuda()


        # (batch_size, n_embeddings)
        center = net([center_one, center_second])
        onehop = net([one_one, one_second])

        features = torch.cat([center,onehop], dim=1)
        outputs = classifier(features)
        loss = criterion(outputs, targets)

        acc, p, r, ratio = accuracy(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip != -1:
            clip_grad_norm_(params, args.grad_clip)
        optimizer.step()

        losses.update(loss.item(), center_one.shape[0])
        accs.update(acc, center_one.shape[0])
        precisions.update(p, center_one.shape[0])
        recalls.update(r, center_one.shape[0])
        ratios.update(ratio, center_one.shape[0])
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if i % args.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {losses.val:.3f} ({losses.avg:.3f})\t'
                'pos {ratios.val:.3f} ({ratios.avg:.3f})\t'
                'Accuracy {accuracy.val:.3f} ({accuracy.avg:.3f})\t'
                'Precison {precisions.val:.3f} ({precisions.avg:.3f})\t'
                'Recall {recalls.val:.3f} ({recalls.avg:.3f})'
                .format(
                    epoch, i+step, N, batch_time=batch_time,
                    losses=losses, ratios=ratios,
                    accuracy=accs, precisions=precisions, recalls=recalls))
        if i % args.save_step == 0:
            save_checkpoint({
                'step': i,
                'epoch': epoch,
                'model': [net.state_dict(), classifier.state_dict()],
                'args': args,
            }, fpath=os.path.join(args.logs_dir, 'epoch_{}_step_{}.ckpt'.format(epoch,i)))



def parse():

    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_path', default='/disk2/zhangqi/emore/feats/labeled_feats.npz',
                        help='path to feats')
    parser.add_argument('--knn_path', default='/disk2/zhangqi/emore/knn_200/labeled_knn.npy',
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
    parser.add_argument('--workers', default=4, type=int,
                        help='Number of 1ata loader workers.')
    parser.add_argument('--save_step', default=5000, type=int,
                        help='Number of steps to save the model.')
    parser.add_argument('--log_step', default=100, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--logs_dir', default='results_imgs',
                        help='Path to save Tensorboard log.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--sample_hops', default=10, type=int)
    parser.add_argument('--gat', type=int)
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = parse()
    main(args)
