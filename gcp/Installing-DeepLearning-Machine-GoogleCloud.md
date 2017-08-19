Full logical steps to script
===========================


Go to GCP platform:  https://cloud.google.com/

Create your VM and increase / request quota to have GPU support: https://console.cloud.google.com/compute/quotas

Adjust your firewall settings: https://console.cloud.google.com/networking/firewalls/list

Go to isntances: https://console.cloud.google.com/compute/instances

Add GPU: https://cloud.google.com/compute/docs/gpus/add-gpus

Add Script to install Nvidia drivers and CUDA while it cooks your VM

Do : `sudo apt-get update`

Check here if you need more info:  https://www.tensorflow.org/install/install_linux#nvidia_requirements_to_run_tensorflow_with_gpu_support

Get cuDNN:  https://developer.nvidia.com/rdp/form/cudnn-download-survey

Download cuDNN: https://developer.nvidia.com/cudnn

Get Anaconda : 

Build TensorFlor from source (AVX, SSE support): https://alliseesolutions.wordpress.com/2016/09/08/install-gpu-tensorflow-from-sources-w-ubuntu-16-04-and-cuda-8-0-rc/

Do this:  `sudo apt-get install openjdk-8-jdk git python-dev python3-dev python-numpy python3-numpy python-six python3-six build-essential python-pip python3-pip python-virtualenv swig python-wheel python3-wheel libcurl3-dev libcupti-dev`

Unpack: `tar -xzvf cudnn-8.0-linux-x64-v5.1.tgz`

Copy these and change mod: 
`sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*`


vim or nano the file:  `nano ~/.bashrc`

and add the following lines: 

`export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda`

Refresh:  `source ~/.bashrc`

Echo:  `echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list`

curl:  `curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -`

Do:  `sudo apt-get update`

Gate Bazel:  `sudo apt-get install bazel`

Update:  `sudo apt-get upgrade bazel`

Get Tensorflow:  `git clone https://github.com/tensorflow/tensorflow`

 `cd ~/tensorflow`

`git checkout r1.2`

 `./configure`

Add these params!

```
bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package

bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

sudo pip3 install /tmp/tensorflow_pkg/tensorflow[PRESS TAB TO COMPLETE FILENAME]
