import mxnet
import mxnet as mx
import numpy as np
import torch
from mxnet import recordio
from torch.utils.data import Dataset
from torchvision import transforms

from config import path_imgidx, path_imgrec, num_workers, num_samples

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}


class ArcFaceDataset(Dataset):
    def __init__(self, split):
        self.imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')

        if split == 'train':
            self.transformer = data_transforms['train']

    def __getitem__(self, i):
        try:
            header, s = recordio.unpack(self.imgrec.read_idx(i + 1))
        except mxnet.base.MXNetError:
            print(i)
            raise
        img = mx.image.imdecode(s).asnumpy()

        class_id = int(header.label)

        img = transforms.ToPILImage()(img)
        img = self.transformer(img)

        return img, class_id

    def __len__(self):
        return num_samples

    def shuffle(self):
        np.random.shuffle(self.samples)


if __name__ == "__main__":
    train_dataset = ArcFaceDataset('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True,
                                               num_workers=num_workers,
                                               pin_memory=True)

    print(len(train_dataset))
    print(len(train_loader))
