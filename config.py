import os

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

# Model parameters
image_w = 112
image_h = 112
channel = 3

# Training parameters
num_workers = 1  # for data-loading; right now, only 1 works with h5py
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 100  # print training/validation stats every __ batches
checkpoint = None  # path to checkpoint, None if none

# Data parameters
num_classes = 85164
num_samples = 3804847
DATA_DIR = 'data'
IMG_DIR = 'data/faces_ms1m_112x112'
path_imgidx = os.path.join(IMG_DIR, 'train.idx')
path_imgrec = os.path.join(IMG_DIR, 'train.rec')
