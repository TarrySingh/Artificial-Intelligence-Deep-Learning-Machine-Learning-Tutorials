#duplicate.sh
source=BigGAN_I128_hdf5_seed0_Gch64_Dch64_bs256_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitxavier_Dinitxavier_Gshared_alex0
target=BigGAN_I128_hdf5_seed0_Gch64_Dch64_bs256_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitxavier_Dinitxavier_Gshared_alex0A
logs_root=logs
weights_root=weights
echo "copying ${source} to ${target}"
cp -r ${logs_root}/${source} ${logs_root}/${target}
cp ${logs_root}/${source}_log.jsonl ${logs_root}/${target}_log.jsonl
cp ${weights_root}/${source}_G.pth ${weights_root}/${target}_G.pth
cp ${weights_root}/${source}_G_ema.pth ${weights_root}/${target}_G_ema.pth
cp ${weights_root}/${source}_D.pth ${weights_root}/${target}_D.pth
cp ${weights_root}/${source}_G_optim.pth ${weights_root}/${target}_G_optim.pth
cp ${weights_root}/${source}_D_optim.pth ${weights_root}/${target}_D_optim.pth
cp ${weights_root}/${source}_state_dict.pth ${weights_root}/${target}_state_dict.pth