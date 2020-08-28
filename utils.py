from datetime import datetime
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import io
from torchvision import transforms as trans
from data.data_pipe import de_preprocess
import torch
from model import l2_norm
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
 
 
 
def separate_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])
    return paras_only_bn, paras_wo_bn
 
def prepare_facebank(conf, model, mtcnn, tta = True):
    model.eval()
    embeddings =  []
    names = ['Unknown']
    for path in conf.facebank_path.iterdir():
        if path.is_file():
            continue
        else:
            embs = []
            for file in path.iterdir():
                if not file.is_file():
                    continue
                else:
                    try:
                        img = Image.open(file)
                    except:
                        continue
                    if img.size != (112, 112):
                        img = mtcnn.align(img)
                    with torch.no_grad():
                        if tta:
                            mirror = trans.functional.hflip(img)
                            emb = model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                            emb_mirror = model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                            embs.append(l2_norm(emb + emb_mirror))
                        else:                       
                            embs.append(model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
        if len(embs) == 0:
            continue
        embedding = torch.cat(embs).mean(0,keepdim=True)
        embeddings.append(embedding)
        names.append(path.name)
    embeddings = torch.cat(embeddings)
    names = np.array(names)
    torch.save(embeddings, conf.facebank_path/'facebank.pth')
    np.save(conf.facebank_path/'names', names)
    return embeddings, names
 
def load_facebank(conf):
    embeddings = torch.load(conf.facebank_path/'facebank.pth')
    names = np.load(conf.facebank_path/'names.npy')
    return embeddings, names
 
def face_reader(conf, conn, flag, boxes_arr, result_arr, learner, mtcnn, targets, tta):
    while True:
        try:
            image = conn.recv()
        except:
            continue
        try:           
            bboxes, faces = mtcnn.align_multi(image, limit=conf.face_limit)
        except:
            bboxes = []
             
        results = learner.infer(conf, faces, targets, tta)
         
        if len(bboxes) > 0:
            print('bboxes in reader : {}'.format(bboxes))
            bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
            bboxes = bboxes.astype(int)
            bboxes = bboxes + [-1,-1,1,1] # personal choice           
            assert bboxes.shape[0] == results.shape[0],'bbox and faces number not same'
            bboxes = bboxes.reshape([-1])
            for i in range(len(boxes_arr)):
                if i < len(bboxes):
                    boxes_arr[i] = bboxes[i]
                else:
                    boxes_arr[i] = 0
            for i in range(len(result_arr)):
                if i < len(results):
                    result_arr[i] = results[i]
                else:
                    result_arr[i] = -1
        else:
            for i in range(len(boxes_arr)):
                boxes_arr[i] = 0 # by default,it's all 0
            for i in range(len(result_arr)):
                result_arr[i] = -1 # by default,it's all -1
        print('boxes_arr ： {}'.format(boxes_arr[:4]))
        print('result_arr ： {}'.format(result_arr[:4]))
        flag.value = 0
 
hflip = trans.Compose([
            de_preprocess,
            trans.ToPILImage(),
            trans.functional.hflip,
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
 
def hflip_batch(imgs_tensor):
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)
    return hfliped_imgs
 
def get_time():
    return (str(datetime.now())[:-10]).replace(' ','-').replace(':','-')
 
def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)
    plot = plt.plot(fpr, tpr, linewidth=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()
    return buf
 
def draw_box_name(bbox,name,frame):
    frame = cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),6)
    frame = cv2.putText(frame,
                    name,
                    (bbox[0],bbox[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0,255,0),
                    3,
                    cv2.LINE_AA)
    return frame
 
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
 
 
class GivenSizeSampler(Sampler):
    '''
    Sampler with given total size
    '''
    def __init__(self, dataset, total_size=None, rand_seed=None, sequential=False, silent=False):
        self.rand_seed = rand_seed if rand_seed is not None else 0
        self.dataset = dataset
        self.epoch = 0
        self.sequential = sequential
        self.silent = silent
        self.total_size = total_size if total_size is not None else len(self.dataset)
 
    def __iter__(self):
        # deterministically shuffle based on epoch
        if not self.sequential:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.rand_seed)
            origin_indices = list(torch.randperm(len(self.dataset), generator=g))
        else:
            origin_indices = list(range(len(self.dataset)))
        indices = origin_indices[:]
 
        # add extra samples to meet self.total_size
        extra = self.total_size - len(origin_indices)
        if not self.silent:
            print('Origin Size: {}\tAligned Size: {}'.format(len(origin_indices), self.total_size))
        if extra < 0:
            indices = indices[:self.total_size]
        while extra > 0:
            intake = min(len(origin_indices), extra)
            indices += origin_indices[:intake]
            extra -= intake
        assert len(indices) == self.total_size, "{} vs {}".format(len(indices), self.total_size)
 
        return iter(indices)
 
    def __len__(self):
        return self.total_size
 
    def set_epoch(self, epoch):
        self.epoch = epoch
