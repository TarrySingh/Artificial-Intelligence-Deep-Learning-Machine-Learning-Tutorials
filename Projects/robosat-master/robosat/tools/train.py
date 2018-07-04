import os
import sys
import argparse
import collections

from PIL import Image

import torch
import torch.backends.cudnn
from torch.nn import DataParallel
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize

from tqdm import tqdm

from robosat.transforms import MaskToTensor, ConvertImageMode, ImageToTensor
from robosat.datasets import SlippyMapTilesConcatenation
from robosat.metrics import MeanIoU
from robosat.losses import CrossEntropyLoss2d
from robosat.unet import UNet
from robosat.utils import plot
from robosat.config import load_config


def add_parser(subparser):
    parser = subparser.add_parser('train', help='trains model on dataset',
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model', type=str, required=True, help='path to model configuration file')
    parser.add_argument('--dataset', type=str, required=True, help='path to dataset configuration file')
    parser.add_argument('--resume', type=str, required=False, help='checkpoint to resume training from')

    parser.set_defaults(func=main)


def main(args):
    model = load_config(args.model)
    dataset = load_config(args.dataset)

    device = torch.device('cuda' if model['common']['cuda'] else 'cpu')

    if model['common']['cuda'] and not torch.cuda.is_available():
        sys.exit('Error: CUDA requested but not available')

    # if args.batch_size < 2:
    #     sys.exit('Error: PSPNet requires more than one image for BatchNorm in Pyramid Pooling')

    os.makedirs(model['common']['checkpoint'], exist_ok=True)

    num_classes = len(dataset['common']['classes'])
    net = UNet(num_classes).to(device)

    if args.resume:
        path = os.path.join(model['common']['checkpoint'], args.resume)

        cuda = model['common']['cuda']

        def map_location(storage, _):
            return storage.cuda() if cuda else storage.cpu()

        chkpt = torch.load(path, map_location=map_location)
        net.load_state_dict(chkpt)
        resume_at_epoch = int(args.resume[11:16])
    else:
        resume_at_epoch = 0

    if model['common']['cuda']:
        torch.backends.cudnn.benchmark = True
        net = DataParallel(net)

    optimizer = SGD(net.parameters(), lr=model['opt']['lr'], momentum=model['opt']['momentum'])

    scheduler = MultiStepLR(optimizer, milestones=model['opt']['milestones'], gamma=model['opt']['gamma'])

    weight = torch.Tensor(dataset['weights']['values'])

    for i in range(resume_at_epoch):
        scheduler.step()

    criterion = CrossEntropyLoss2d(weight=weight).to(device)
    # criterion = FocalLoss2d(weight=weight).to(device)

    train_loader, val_loader = get_dataset_loaders(model, dataset)

    num_epochs = model['opt']['epochs']

    history = collections.defaultdict(list)

    for epoch in range(resume_at_epoch, num_epochs):
        print('Epoch: {}/{}'.format(epoch + 1, num_epochs))

        train_hist = train(train_loader, num_classes, device, net, optimizer, scheduler, criterion)
        print('Train loss: {:.4f}, mean IoU: {:.4f}'.format(train_hist['loss'], train_hist['iou']))

        for k, v in train_hist.items():
            history['train ' + k].append(v)

        val_hist = validate(val_loader, num_classes, device, net, criterion)
        print('Validate loss: {:.4f}, mean IoU: {:.4f}'.format(val_hist['loss'], val_hist['iou']))

        for k, v in val_hist.items():
            history['val ' + k].append(v)

        visual = 'history-{:05d}-of-{:05d}.png'.format(epoch + 1, num_epochs)
        plot(os.path.join(model['common']['checkpoint'], visual), history)

        checkpoint = 'checkpoint-{:05d}-of-{:05d}.pth'.format(epoch + 1, num_epochs)
        torch.save(net.state_dict(), os.path.join(model['common']['checkpoint'], checkpoint))


def train(loader, num_classes, device, net, optimizer, scheduler, criterion):
    num_samples = 0
    running_loss = 0

    iou = MeanIoU(range(num_classes))

    net.train()
    scheduler.step()

    for images, masks, tiles in tqdm(loader, desc='Train', unit='batch', ascii=True):
        images = images.to(device)
        masks = masks.to(device)

        assert images.size()[2:] == masks.size()[1:], 'resolutions for images and masks are in sync'

        num_samples += int(images.size(0))

        optimizer.zero_grad()
        outputs = net(images)

        assert outputs.size()[2:] == masks.size()[1:], 'resolutions for predictions and masks are in sync'
        assert outputs.size()[1] == num_classes, 'classes for predictions and dataset are in sync'

        loss = criterion(outputs, masks)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        for mask, output in zip(masks, outputs):
            mask = mask.data.cpu().numpy()
            prediction = output.data.max(0)[1].cpu().numpy()
            iou.add(mask.ravel(), prediction.ravel())

    assert num_samples > 0, 'dataset contains training images and labels'

    return {'loss': running_loss / num_samples, 'iou': iou.get()}


def validate(loader, num_classes, device, net, criterion):
    num_samples = 0
    running_loss = 0

    iou = MeanIoU(range(num_classes))

    net.eval()

    for images, masks, tiles in tqdm(loader, desc='Validate', unit='batch', ascii=True):
        images = images.to(device)
        masks = masks.to(device)

        assert images.size()[2:] == masks.size()[1:], 'resolutions for images and masks are in sync'

        num_samples += int(images.size(0))

        outputs = net(images)

        assert outputs.size()[2:] == masks.size()[1:], 'resolutions for predictions and masks are in sync'
        assert outputs.size()[1] == num_classes, 'classes for predictions and dataset are in sync'

        loss = criterion(outputs, masks)

        running_loss += loss.item()

        for mask, output in zip(masks, outputs):
            mask = mask.data.cpu().numpy()
            prediction = output.data.max(0)[1].cpu().numpy()
            iou.add(mask.ravel(), prediction.ravel())

    assert num_samples > 0, 'dataset contains validation images and labels'

    return {'loss': running_loss / num_samples, 'iou': iou.get()}


def get_dataset_loaders(model, dataset):
    target_size = (model['common']['image_size'], ) * 2
    batch_size = model['common']['batch_size']
    path = dataset['common']['dataset']

    mean, std = dataset['stats']['mean'], dataset['stats']['std']

    image_transform = Compose([
        ConvertImageMode('RGB'),
        Resize(target_size, Image.BILINEAR),
        CenterCrop(target_size),
        ImageToTensor(),
        Normalize(mean=mean, std=std)])

    target_transform = Compose([
        ConvertImageMode('P'),
        Resize(target_size, Image.NEAREST),
        CenterCrop(target_size),
        MaskToTensor()])

    train_dataset = SlippyMapTilesConcatenation(
        [os.path.join(path, 'training', 'images')],
        [image_transform],
        os.path.join(path, 'training', 'labels'),
        target_transform)

    val_dataset = SlippyMapTilesConcatenation(
        [os.path.join(path, 'validation', 'images')],
        [image_transform],
        os.path.join(path, 'validation', 'labels'),
        target_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)

    return train_loader, val_loader
