from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import transforms as trans
from torchvision.datasets import ImageFolder, DatasetFolder
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import torch
import mxnet as mx
import cv2
import bcolz
import pickle
from tqdm import tqdm
import os
import sys
from torch.utils.data import Sampler


starts = [0, 584013, 1164672, 1740301, 2314488, 2890517, 3465678, 4046365, 4628523]

def de_preprocess(tensor):
    return tensor*0.5 + 0.5
    
def get_train_dataset(imgs_folder):
    train_transform = trans.Compose([
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    ds = ImageFolder(str(imgs_folder), train_transform)
    class_num = ds[-1][1] + 1
    print('loading train dataset:{} done'.format(class_num))
    return ds, class_num

class SubsetSampler(Sampler):
    r"""Samples elements from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)



def get_train_loader(conf):
    if conf.data_mode in ['ms1m', 'concat']:
        ms1m_ds, ms1m_class_num = get_train_dataset(conf.ms1m_folder+'/imgs')
        print('ms1m loader generated')
    if conf.data_mode in ['vgg', 'concat']:
        vgg_ds, vgg_class_num = get_train_dataset(conf.vgg_folder+'/imgs')
        print('vgg loader generated')        
    if conf.data_mode == 'vgg':
        ds = vgg_ds
        class_num = vgg_class_num
    elif conf.data_mode == 'ms1m':
        ds = ms1m_ds
        class_num = ms1m_class_num
    elif conf.data_mode == 'concat':
        for i,(url,label) in enumerate(vgg_ds.imgs):
            vgg_ds.imgs[i] = (url, label + ms1m_class_num)
        ds = ConcatDataset([ms1m_ds,vgg_ds])
        class_num = vgg_class_num + ms1m_class_num
    elif conf.data_mode == 'emore':
        ds, class_num = get_train_dataset(conf.emore_folder)
    loader = DataLoader(ds, batch_size=conf.batch_size, shuffle=True, pin_memory=conf.pin_memory, num_workers=conf.num_workers)
    return loader, class_num 



def get_joint_loader(conf):
    train_transform = trans.Compose([
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    ds = JointDataset(str(conf.emore_folder), conf.num_splits, train_transform)
    loader = DataLoader(ds, batch_size=conf.batch_size, shuffle=True, pin_memory=conf.pin_memory, num_workers=conf.num_workers)
    class_num = ds.class_num
    return loader, class_num


    
def load_bin(path, rootdir, transform, image_size=[112,112]):
    if not rootdir.exists():
        rootdir.mkdir()
    bins, issame_list = pickle.load(open(str(path), 'rb'), encoding='bytes')
    data = bcolz.fill([len(bins), 3, image_size[0], image_size[1]], dtype=np.float32, rootdir=str(rootdir), mode='w')
    for i in range(len(bins)):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(img.astype(np.uint8))
        data[i, ...] = transform(img)
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
    print(data.shape)
    np.save(str(rootdir)+'_list', np.array(issame_list))
    return data, issame_list

def get_val_pair(path, name):
    carray = bcolz.carray(rootdir = os.path.join(path,name), mode='r')
    issame = np.load(path+'/{}_list.npy'.format(name))
    return carray, issame

def get_val_data(data_path):
    agedb_30, agedb_30_issame = get_val_pair(data_path, 'agedb_30')
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_fp')
    lfw, lfw_issame = get_val_pair(data_path, 'lfw')
    return agedb_30, cfp_fp, lfw, agedb_30_issame, cfp_fp_issame, lfw_issame

def load_mx_rec(rec_path):
    save_path = rec_path/'imgs'
    if not save_path.exists():
        save_path.mkdir()
    imgrec = mx.recordio.MXIndexedRecordIO(str(rec_path/'train.idx'), str(rec_path/'train.rec'), 'r')
    img_info = imgrec.read_idx(0)
    header,_ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    for idx in tqdm(range(1,max_idx)):
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        label = int(header.label)
        img = Image.fromarray(img)
        label_path = save_path/str(label)
        if not label_path.exists():
            label_path.mkdir()
        img.save(label_path/'{}.jpg'.format(idx), quality=95)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def make_dataset(dir, class_to_idx, pseudo_classes, remove_single=True, extensions=None, is_valid_file=None):

    images = []
    dir = os.path.expanduser(dir)


    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    flag = 0
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    #item = (path, class_to_idx[target])
                    if remove_single and pseudo_classes[flag] == -1:
                        flag += 1
                        continue
                    item = (path, pseudo_classes[flag])
                    flag += 1
                    images.append(item)
    return images
    

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)

class JointDataset(DatasetFolder):
    def __init__(self, root, num_splits=1, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        super(JointDataset, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform)
        ds, num = get_train_dataset(root)
        self.samples = ds.samples
        for i in range(num_splits):
            ds, nuM = get_train_dataset(root.replace('trainset','testset/split_{}'.format(i+1)))
            #self.samples.extend(ds)
            self.samples.extend([(a,b+num) for a,b in ds.samples])
            num += nuM
        self.targets = [s[1] for s in self.samples]
        self.class_num = len(set(self.targets))
        print('samples num: {}, class_num: {}'.format(len(self.samples), self.class_num))
    
    
class JointPseudoDataset(DatasetFolder):
    def __init__(self, root, pseudo_classes, remove_single=True, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        super(JointPseudoDataset, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform)
        ds, num = get_train_dataset(root)
        self.samples = ds.samples
        for i in range(len(pseudo_classes)):
            ds, nuM = get_pseudo_dataset((root.replace('trainset','testset'),i+1), pseudo_classes[i], remove_single)
            self.samples.extend([(a,b+num) for a,b in ds.samples])
            num += nuM
        self.targets = [s[1] for s in self.samples]
        self.class_num = len(set(self.targets))
        print('samples num: {}, class_num: {}'.format(len(self.samples), self.class_num))



class PSEUDO(DatasetFolder):
    def __init__(self, remove_single, root, pseudo_classes, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        super(PSEUDO, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform)
    
        self.remove_single = remove_single
        classes, class_to_idx = self._find_classes(self.root)
        self.samples = make_dataset(self.root, class_to_idx, pseudo_classes, self.remove_single, self.extensions, is_valid_file)
        self.targets = [s[1] for s in self.samples]
        self.class_num = len(set(self.targets))


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns=0:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


    def __len__(self):
        return len(self.samples)
    
    
    def _find_classes(self, dir):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx



def get_pseudo_dataset(imgs_folder, pseudo_classes, remove_single):
    split_index = imgs_folder[1]
    train_transform = trans.Compose([
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    ds = PSEUDO(remove_single, os.path.join(str(imgs_folder[0]),'split_{}'.format(split_index)), pseudo_classes, train_transform)
    class_num = ds.class_num
    print('loading pseudo {}: {} done'.format(split_index, class_num))
    return ds, class_num
