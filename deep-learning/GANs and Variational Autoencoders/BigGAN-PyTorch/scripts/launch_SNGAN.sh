#!/bin/bash
python train.py \
--dataset I128_hdf5 --parallel --shuffle  --num_workers 8 --batch_size 64  \
--num_G_accumulations 1 --num_D_accumulations 1 \
--num_D_steps 5 --G_lr 2e-4 --D_lr 2e-4 --D_B2 0.900 --G_B2 0.900 \
--G_attn 0 --D_attn 0 \
--G_nl relu --D_nl relu \
--SN_eps 1e-8 --BN_eps 1e-5 --adam_eps 1e-8 \
--G_ortho 0.0 \
--D_thin \
--G_init xavier --D_init xavier \
 --G_eval_mode \
--test_every 2000 --save_every 1000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--name_suffix SNGAN \