# use z_var to change the variance of z for all the sampling
# use --mybn --accumulate_stats --num_standing_accumulations 32 to 
# use running stats
python sample.py \
--dataset I128_hdf5 --parallel --shuffle  --num_workers 8 --batch_size 256  \
--num_G_accumulations 8 --num_D_accumulations 8 \
--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 64 --D_attn 64 \
--G_ch 96 --D_ch 96 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared \
--G_init ortho --D_init ortho --skip_init \
--hier --dim_z 120 --shared_dim 128 \
--ema --ema_start 20000 \
--use_multiepoch_sampler \
--test_every 2000 --save_every 1000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--skip_init --G_batch_size 512  --use_ema --G_eval_mode --sample_trunc_curves 0.05_0.05_1.0  \
--sample_inception_metrics --sample_npz  --sample_random --sample_sheets --sample_interps
