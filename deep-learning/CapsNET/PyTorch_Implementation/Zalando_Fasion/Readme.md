# Dynamic Routing Between Capsules with Zalando Fashion MNIST Data

[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=plastic)](CONTRIBUTING.md)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=plastic)](https://opensource.org/licenses/Apache-2.0)
![completion](https://img.shields.io/badge/completion%20state-70%25-blue.svg?style=plastic)

This ia a barebone CUDA-enabled Torch / PyTorch with Facebook's Visdom Server implementation of the CapsNet architecture in the paper "Dynamic Routing Between Capsules" by [Kenta Iwasaki](https://github.com/iwasaki-kenta) on behalf of Gram.AI.

Download Torchnet (you can do it via your Jupyter notebook) as the training for the model is done using [TorchNet](https://github.com/pytorch/tnt), with MNIST dataset loading and preprocessing done with [TorchVision](https://github.com/pytorch/vision).

## Description

> A capsule is a group of neurons whose activity vector represents the instantiation parameters of a specific type of entity such as an object or object part. We use the length of the activity vector to represent the probability that the entity exists and its orientation to represent the instantiation paramters. Active capsules at one level make predictions, via transformation matrices, for the instantiation parameters of higher-level capsules. When multiple predictions agree, a higher level capsule becomes active. We show that a discrimininatively trained, multi-layer capsule system achieves state-of-the-art performance on MNIST and is considerably better than a convolutional net at recognizing highly overlapping digits. To achieve these results we use an iterative routing-by-agreement mechanism: A lower-level capsule prefers to send its output to higher level capsules whose activity vectors have a big scalar product with the prediction coming from the lower-level capsule.

**Routing Algorithm**


<img src="https://github.com/TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials/blob//master/deep-learning/CapsNET/PyTorch_Implementation/images/routing_algorithm.png">

**Margin Loss for Digita Existence**

<a href="https://www.codecogs.com/eqnedit.php?latex=L_c&space;=&space;T_c&space;max(0,&space;m^&plus;&space;-&space;||v_c||)^2&space;&plus;&space;\lambda&space;(1&space;-&space;T_c)&space;max(0,&space;||v_c||&space;-&space;m^-)^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L_c&space;=&space;T_c&space;max(0,&space;m^&plus;&space;-&space;||v_c||)^2&space;&plus;&space;\lambda&space;(1&space;-&space;T_c)&space;max(0,&space;||v_c||&space;-&space;m^-)^2" title="L_c = T_c max(0, m^+ - ||v_c||)^2 + \lambda (1 - T_c) max(0, ||v_c|| - m^-)^2" /></a>

**Architechture Layer**

A 3-layer CapsNet architecture.

<img src="https://github.com/TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials/blob/master/deep-learning/CapsNET/PyTorch_Implementation/images/architecture.png">

**Decoder Layer**

<img src="https://github.com/TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials/blob/master/deep-learning/CapsNET/PyTorch_Implementation/images/decoder_layer.png">

Paper written by Sara Sabour, Nicholas Frosst, and Geoffrey E. Hinton. For more information, please check out the paper [here](https://arxiv.org/abs/1710.09829).

__Note__: Affine-transformations for the data augmentation stage have not been implemented yet. This implementation only provides an efficient implementation for the dynamic routing procedure, example CapsNet architecture, and squashing functions mentioned in the paper.

## Requirements

* Python 3 (Mine is 3.6.x distro)
* PyTorch
* TorchVision
* TorchNet (Download it via your Jupyter Notebook, easy peasy :))
* TQDM (for checking out the progress)
* Visdom (Neat visualization of your train, acuracy and margin losses)

## Usage

**Step 1** Adjust the "hyperparamters" in the cell block and play with the results such as number of training epochs, batch sizes, etc. inside `capsule_network.py`.

```python
batch_size = 128
num_classes = 10
num_epochs = 200 
num_rout_iter = 3 (Number of routing iterations)
```
I will be adding other hyperparameters such as learning rate and see if this helps improve the results.

Follow the simple instructions in this Jupyter Notebook.

## Benchmarks

Highest accuracy was [**coming soon as I'm still running it**] after 200 epochs on request of the Zalando Researcher [@HanXiao](https://github.com/hanxiao). 
This image is after just 10 epochs, we're doing 200 now. 

Accuracy is climbing slowly but this is expected.

I will go for 2000 later [If I have time obviously!!!] if there is more request or need to investigate and see if it improves.

**10 Epochs**

![Training progress after 25 Epochs.](images/10.png)

**200 Epochs**

![Training progress after 100 Epochs.](images/200.png)

**1000 Epochs**

Trying out these changes...

```python
batch_size = 256
num_classes = 10
num_epochs = 1000 
num_rout_iter = 3 (Number of routing iterations)
```
-- Note: Since I doubled the batch_size, num inter/se have doubled to ~1.4 it/s. Let's see if these changes in hyperparameters and batch_size affect the accuracy and loss.

---> 438 Epochs later

![Training progress after 1000 Epochs.](images/438.png)

![Training progress after 1000 Epochs.](images/comingsoon.jpg)

**2000 Epochs**

![Training progress after 2000 Epochs.](images/comingsoon.jpg)

Default PyTorch Adam optimizer hyperparameters were used with no learning rate scheduling. Epochs with batch size of 128 is taking me approximately  takes ~2.85 interations per minute OR approx. ~ 2.5 mins per epoch on an Ubuntu 16.04 Server with 64GB RAM, 8 Gig GTX 1080 GPU & (2 x 500GB) 1TB SSD Disk Space. 

## TODO

* Extension to other datasets apart from MNIST.
* CIFAR10 and other datasets
* Images larger that the MNIST and CIFAR (ImageNet etc)
* More Complex Data Structures Such as Amazon Forests etc.

## Credits & Sources

Primarily referenced these two TensorFlow and Keras implementations:
1. [Keras implementation by @XifengGuo](https://github.com/XifengGuo/CapsNet-Keras)
2. [TensorFlow implementation by @naturomics](https://github.com/naturomics/CapsNet-Tensorflow)
3. [Kenta Iwasaki for making this PyTorch implementation](https://github.com/iwasaki-kenta)

Also worthwhile following a bunch of discussions here such as this: [@InnerPeace-Wu](https://github.com/InnerPeace-Wu) for a [discussion on the dynamic routing procedure](https://github.com/XifengGuo/CapsNet-Keras/issues/1) outlined in the paper.

**NOTE**: I will be rolling all these up to the main README.md file for the CapsNets detailing about what I learned from all the three implementations of Keras, PyTorch and Tensorflow. so stay tuned!
## Contact

I am testing a whole lot of new and upcoming AI and Data Models. Feel free to reach out to me if you need help or suggestions.
Here is my contact detail at [LinkedIn](https://www.linkedin.com/in/tarrysingh/), [Quora](https://www.quora.com/profile/Tarry-Singh) , [My Website - TarrySingh.com](https://tarrysingh.com) or simply [email](tarry.singh@gmail.com)
