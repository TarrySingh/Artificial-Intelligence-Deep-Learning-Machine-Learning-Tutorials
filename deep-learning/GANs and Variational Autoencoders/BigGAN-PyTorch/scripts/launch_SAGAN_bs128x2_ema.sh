#!/bin/bash
python train.py \
--dataset I128_hdf5 --parallel --shuffle  --num_workers 8 --batch_size 128  \
--num_G_accumulations 2 --num_D_accumulations 2 \
--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 64 --D_attn 64 \
--G_nl relu --D_nl relu \
--SN_eps 1e-8 --BN_eps 1e-5 --adam_eps 1e-8 \
--G_ortho 0.0 \
--G_init xavier --D_init xavier \
--ema --use_ema --ema_start 2000 --G_eval_mode \
--test_every 2000 --save_every 1000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--name_suffix SAGAN_ema \