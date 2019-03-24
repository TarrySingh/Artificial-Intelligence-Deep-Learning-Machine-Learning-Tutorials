""" Convert dataset to HDF5
    This script preprocesses a dataset and saves it (images and labels) to 
    an HDF5 file for improved I/O. """
import os
import sys
from argparse import ArgumentParser
from tqdm import tqdm, trange
import h5py as h5

import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import utils

def prepare_parser():
  usage = 'Parser for ImageNet HDF5 scripts.'
  parser = ArgumentParser(description=usage)
  parser.add_argument(
    '--dataset', type=str, default='I128',
    help='Which Dataset to train on, out of I128, I256, C10, C100;'
         'Append "_hdf5" to use the hdf5 version for ISLVRC (default: %(default)s)')
  parser.add_argument(
    '--dataset_root', type=str, default='data',
    help='Default location where data is stored (default: %(default)s)')
  parser.add_argument(
    '--batch_size', type=int, default=256,
    help='Default overall batchsize (default: %(default)s)')
  parser.add_argument(
    '--num_workers', type=int, default=16,
    help='Number of dataloader workers (default: %(default)s)')
  parser.add_argument(
    '--chunk_size', type=int, default=500,
    help='Default overall batchsize (default: %(default)s)')
  parser.add_argument(
    '--compression', action='store_true', default=False,
    help='Use LZF compression? (default: %(default)s)')
  return parser


def run(config):
  if 'hdf5' in config['dataset']:
    raise ValueError('Reading from an HDF5 file which you will probably be '
                     'about to overwrite! Override this error only if you know '
                     'what you''re doing!')
  # Get image size
  config['image_size'] = utils.imsize_dict[config['dataset']]

  # Update compression entry
  config['compression'] = 'lzf' if config['compression'] else None #No compression; can also use 'lzf' 

  # Get dataset
  kwargs = {'num_workers': config['num_workers'], 'pin_memory': False, 'drop_last': False}
  train_loader = utils.get_data_loaders(dataset=config['dataset'],
                                        batch_size=config['batch_size'],
                                        shuffle=False,
                                        dataset_root=config['dataset_root'],
                                        use_multiepoch_sampler=False,
                                        **kwargs)[0]     

  # HDF5 supports chunking and compression. You may want to experiment 
  # with different chunk sizes to see how it runs on your machines.
  # Chunk Size/compression     Read speed @ 256x256   Read speed @ 128x128  Filesize @ 128x128    Time to write @128x128
  # 1 / None                   20/s
  # 500 / None                 ramps up to 77/s       102/s                 61GB                  23min
  # 500 / LZF                                         8/s                   56GB                  23min
  # 1000 / None                78/s
  # 5000 / None                81/s
  # auto:(125,1,16,32) / None                         11/s                  61GB        

  print('Starting to load %s into an HDF5 file with chunk size %i and compression %s...' % (config['dataset'], config['chunk_size'], config['compression']))
  # Loop over train loader
  for i,(x,y) in enumerate(tqdm(train_loader)):
    # Stick X into the range [0, 255] since it's coming from the train loader
    x = (255 * ((x + 1) / 2.0)).byte().numpy()
    # Numpyify y
    y = y.numpy()
    # If we're on the first batch, prepare the hdf5
    if i==0:
      with h5.File(config['dataset_root'] + '/ILSVRC%i.hdf5' % config['image_size'], 'w') as f:
        print('Producing dataset of len %d' % len(train_loader.dataset))
        imgs_dset = f.create_dataset('imgs', x.shape,dtype='uint8', maxshape=(len(train_loader.dataset), 3, config['image_size'], config['image_size']),
                                     chunks=(config['chunk_size'], 3, config['image_size'], config['image_size']), compression=config['compression']) 
        print('Image chunks chosen as ' + str(imgs_dset.chunks))
        imgs_dset[...] = x
        labels_dset = f.create_dataset('labels', y.shape, dtype='int64', maxshape=(len(train_loader.dataset),), chunks=(config['chunk_size'],), compression=config['compression'])
        print('Label chunks chosen as ' + str(labels_dset.chunks))
        labels_dset[...] = y
    # Else append to the hdf5
    else:
      with h5.File(config['dataset_root'] + '/ILSVRC%i.hdf5' % config['image_size'], 'a') as f:
        f['imgs'].resize(f['imgs'].shape[0] + x.shape[0], axis=0)
        f['imgs'][-x.shape[0]:] = x
        f['labels'].resize(f['labels'].shape[0] + y.shape[0], axis=0)
        f['labels'][-y.shape[0]:] = y


def main():
  # parse command line and run    
  parser = prepare_parser()
  config = vars(parser.parse_args())
  print(config)
  run(config)

if __name__ == '__main__':    
  main()