import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

# Model parameters
image_w = 112
image_h = 112
channel = 3
num_classes = 10575
s = 64
m = 0.5
embedding_size = 512
dropout = 0.5

# Training parameters
start_epoch = 0
max_epoch = 50  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 512
lr = 0.1  # learning rate
weight_decay = 5e-4
train_split = 0.9
num_workers = 4  # for data-loading; right now, only 1 works with h5py
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 100  # print training/validation stats every __ batches
checkpoint = None  # path to checkpoint, None if none

# Data parameters
DATA_DIR = 'data'
IMG_DIR = 'data/CASIA-WebFace'
pickle_file = DATA_DIR + '/' + 'CASIA-WebFace.pkl'
