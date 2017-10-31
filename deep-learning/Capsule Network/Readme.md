# CapsNet with Tensorflow (PyTorch and Keras comnig soon!)

[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=plastic)](CONTRIBUTING.md)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=plastic)](https://opensource.org/licenses/Apache-2.0)
![completion](https://img.shields.io/badge/completion%20state-90%25-blue.svg?style=plastic)

A Tensorflow implementation of CapsNet in Hinton's paper [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)

> - **Note from Zhico if you want to follow discussion on WeChat, not mine! (not Tarry's):**

> 1. The routing algorithm has been found not implemented as the same as the one in the paper!(#8, thanks very much, it's my carelessness) Although the current version works, but not effective as the results in the paper, I've paste out some results of the 'wrong' version(It's really interesting). I'll analyze the reasons of this interesting thing
> 2. I've been working continuously for the last few days, coding, talking, writing docs, so I would like to give myself half a day. I will be back tomorrow, and keep updating this repo
> 3. [Here(知乎)](https://zhihu.com/question/67287444/answer/251460831) is my understanding of the section 4 of the paper (the core part of CapsNet), it might be helpful for understanding the code. Thanks for your focus
> 4. If you find out any problems, please let me know. I will try my best to 'kill' it as quickly as possible.




## Requirements
- Python 3.x (I'm using 3.6.x)
- NumPy
- [Tensorflow](https://github.com/tensorflow/tensorflow) (I'm using 1.3.0, others should work, too)
- tqdm (for showing training progress info)
- Scipy
- Check out the *.py files if you want to play around with the hyperparameters.

## Usage

### Training
**Step 1.** 
Clone this repository with ``git``.

```
$ git clone if you want but I'd rather just download the *.py file and tinker along if I were you!
$ create your CapsNET folder with mkdir capsNET
```

**Step 2.** 
Download [MNIST dataset](http://yann.lecun.com/exdb/mnist/), ``mv`` and extract them into ``data/mnist`` directory.(Be careful the backslash appeared around the curly braces when you copy the ``wget `` command to your terminal, remove it)

```
$ mkdir -p data/mnist
$ wget -c -P data/mnist http://yann.lecun.com/exdb/mnist/{train-images-idx3-ubyte.gz,train-labels-idx1-ubyte.gz,t10k-images-idx3-ubyte.gz,t10k-labels-idx1-ubyte.gz}
$ gunzip data/mnist/*.gz
```

**Step 3.** 
Start training with command line:
```
$ pip install tqdm  # install it if you haven't installed yes
$ python train.py
```

### Evaluation
```
$ python eval.py --is_training False
```


## Results
Results for the 'wrong' version(Issues #8):

- training loss
![total_loss]() Coming soon!

![margin_loss]() Coming soon!
![reconstruction_loss]() Coming soon!

- test acc
------------
|Epoch|49|51|
|:----:|:----:|:--:|
|test acc|94.69|94.71|

![test_img1](results/test_000.png)
![test_img2](results/test_015.png)
![test_img3](results/test_030.png)
![test_img4](results/test_045.png)
![test_img5](results/test_075.png)

Results after fix Issues #8: 

### TODO:
- Finish the MNIST version of capsNet (progress:20%)
- Do some different experiments for capsNet:
  * Using other datasets such as CIFAR
	* Adjusting model structure
  * Use with PyTorch and Keras Later

- There is [another new paper](https://openreview.net/pdf?id=HJWLfGWRb) about capsules(submitted to ICLR 2018), follow-up.
- Also follow the original Github from [Zhico's here](https://github.com/naturomics/CapsNet-Tensorflow/edit/master/README.md)
