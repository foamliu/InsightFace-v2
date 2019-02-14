import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

# Model parameters
image_w = 112
image_h = 112
channel = 3
num_classes = 10575
num_samples = 494414

# Training parameters
train_split = 0.9
num_workers = 4  # for data-loading; right now, only 1 works with h5py
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 100  # print training/validation stats every __ batches
checkpoint = None  # path to checkpoint, None if none

# Data parameters
DATA_DIR = 'data'
IMG_DIR = 'data/CASIA-WebFace'
pickle_file = DATA_DIR + '/' + 'CASIA-WebFace.pkl'
