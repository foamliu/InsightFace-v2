import math
import os
from shutil import copyfile

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from config import device, grad_clip, print_freq
from data_gen import ArcFaceDataset
from focal_loss import FocalLoss
from lfw_eval import lfw_test
from mobilenet_v2 import MobileNetV2
from models import resnet18, resnet34, resnet50, resnet101, resnet152, ArcMarginModel
from utils import parse_args, save_checkpoint, AverageMeter, clip_gradient, accuracy, get_logger


def full_log(epoch):
    full_log_dir = 'data/full_log'
    if not os.path.isdir(full_log_dir):
        os.mkdir(full_log_dir)
    filename = 'angles_{}.txt'.format(epoch)
    dst_file = os.path.join(full_log_dir, filename)
    src_file = 'data/angles.txt'
    copyfile(src_file, dst_file)


def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_acc = float('-inf')
    writer = SummaryWriter()
    epochs_since_improvement = 0

    # Initialize / load checkpoint
    if checkpoint is None:
        if args.network == 'r18':
            model = resnet18(args)
        elif args.network == 'r34':
            model = resnet34(args)
        elif args.network == 'r50':
            model = resnet50(args)
        elif args.network == 'r101':
            model = resnet101(args)
        elif args.network == 'r152':
            model = resnet152(args)
        elif args.network == 'mobile':
            model = MobileNetV2()
        else:
            raise TypeError('network {} is not supported.'.format(args.network))

        # print(model)
        model = nn.DataParallel(model)
        metric_fc = ArcMarginModel(args)
        metric_fc = nn.DataParallel(metric_fc)

        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                        lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                         lr=args.lr, weight_decay=args.weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model']
        metric_fc = checkpoint['metric_fc']
        optimizer = checkpoint['optimizer']

    logger = get_logger()

    # Move to GPU, if available
    model = model.to(device)
    metric_fc = metric_fc.to(device)

    # Loss function
    if args.focal_loss:
        criterion = FocalLoss(gamma=args.gamma).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    train_dataset = ArcFaceDataset('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=0.1)

    # Epochs
    for epoch in range(start_epoch, args.end_epoch):
        # One epoch's training
        train_loss, train_acc = train(train_loader=train_loader,
                                      model=model,
                                      metric_fc=metric_fc,
                                      criterion=criterion,
                                      optimizer=optimizer,
                                      epoch=epoch,
                                      logger=logger)

        writer.add_scalar('model/train_loss', train_loss, epoch)
        writer.add_scalar('model/train_acc', train_acc, epoch)

        # One epoch's validation
        lfw_acc, threshold = lfw_test(model)
        writer.add_scalar('model/valid_acc', lfw_acc, epoch)
        writer.add_scalar('model/valid_thres', threshold, epoch)

        # Check if there was an improvement
        is_best = lfw_acc > best_acc
        best_acc = max(lfw_acc, best_acc)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, metric_fc, optimizer, best_acc, is_best)

        scheduler.step(epoch)


def train(train_loader, model, metric_fc, criterion, optimizer, epoch, logger):
    model.train()  # train mode (dropout and batchnorm is used)
    metric_fc.train()

    losses = AverageMeter()
    top5_accs = AverageMeter()

    # Batches
    for i, (img, label) in enumerate(train_loader):
        # Move to GPU, if available
        img = img.to(device)
        label = label.to(device)  # [N, 1]

        # Forward prop.
        feature = model(img)  # embedding => [N, 512]
        output = metric_fc(feature, label)  # class_id_out => [N, 10575]

        # Calculate loss
        loss = criterion(output, label)

        # try:
        #     assert (not math.isnan(loss.item()))
        # except AssertionError:
        #     continue

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        clip_gradient(optimizer, grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())
        top5_accuracy = accuracy(output, label, 5)
        top5_accs.update(top5_accuracy)

        # Print status
        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Top5 Accuracy {top5_accs.val:.3f} ({top5_accs.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                                         loss=losses,
                                                                                         top5_accs=top5_accs))

    return losses.avg, top5_accs.avg


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
