#!/bin/bash
python make_hdf5.py --dataset I128 --batch_size 256 --dataset_root data
python calculate_inception_moments.py --dataset I128_hdf5 --dataset_root data