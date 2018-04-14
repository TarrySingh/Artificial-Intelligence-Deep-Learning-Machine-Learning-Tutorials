# Remove anything linked to nvidia
sudo apt-get remove --purge nvidia*
sudo apt-get autoremove

# Search for your driver
apt search nvidia

# Select one driver (the last one is a decent choice)
sudo apt install nvidia-370

# Test the driver
sudo shutdown -r now
nvidia-smi 

# If it doesn't work, sometimes this is due to a secure boot option of your motherboard, disable it and test again

# Install cuda 
# Get your deb cuda file from https://developer.nvidia.com/cuda-downloads
sudo dpkg -i dev.file
sudo apt update
sudo apt install cuda

# Add cuda to your PATH and install the toolkit
# Also add them to your .bashrc file
export PATH=/usr/local/cuda-9.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda-9.1
nvcc --version

# Use the toolkit to check your CUDA capable devices
cuda-install-samples-9.1.sh ~/.
cd ~/NVIDIA_CUDA-9.1_Samples/1_Utilities/deviceQuery
make
shutdown -r now

# Test cuda
cd ~/NVIDIA_CUDA-9.1_Samples/1_Utilities/deviceQuery
./deviceQuery

# Downloads cudnn deb files from the nvidia website: 
# https://developer.nvidia.com/rdp/cudnn-download
# Install cudnn
tar -zxvf cudnn-9.1-linux-x64-v5.1.tgz 
sudo mv cuda/include/* /usr/local/cuda-9.1/include/.
sudo mv cuda/lib64/* /usr/local/cuda-9.1/lib64/.

# Reload your shell
. ~/.bashrc
