from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset, DataLoader, Sampler
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


