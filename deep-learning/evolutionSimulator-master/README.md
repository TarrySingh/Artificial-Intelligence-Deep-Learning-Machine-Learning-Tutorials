# [DEMO](https://adityathebe.github.io/evolutionSimulator/)
    
# Evolution Simulator [Under Development]

- [x] Neural Network
- [x] Genetic Algorithm
- [x] Physics Environment

# Project Structure

- Environments : Various Environment Models
- Creatures : Various Creature Models
- NeuroEvolution : Neural Network and Genetic Algorithm library
- Lib : p5.js, Matter.js and Tensorflow.js

# System Design

### 1. Neural Network

All creatures have a 3 layer feed-forward Neural Network as their brain. The topology is **4 - 100 - X**, where the number of nodes X in the output layer depend on the number of muscles of the creature. The input data fed to the network are:

- Horizontal velocity
- Vertical Velocity
- Torque
- Height above the ground level

### 2. Genetic Algorithm Design

![](https://visualstudiomagazine.com/articles/2014/03/01/~/media/ECG/visualstudiomagazine/Images/2014/03/EvolutionaryAlgorithm.ashx)

#### a. Score:

A creature can gain points based on the distance it travels from the starting point. The further it travels in the correct direction, the more point it gains. Traveling in the opposite direction, will reduce the point.

#### b. Fitness Function:

The further the creatures go to the right the more they are rewarded.

#### c. Selection Algorithm:

The creatures are selected for breeding based on their fitness value. The fitness value acts like a probability of being chosen for reproduction. Creatures that perform better have higher fitness value and hence has higher chance of reproducing.

#### d. **Crossover:**

*The objective of this function is to generate a new child by combining the genes of two parents*.

Two creatures (*parents*) are selected using the selection algorithm. Their weights are interchanged randomly bit wise as shown in the picture below to form a new set of weights. In our case, a single bit represents a single weight. This new set of weights is used to form a new creature (*child).*

![](https://static.thinkingandcomputing.com/2014/03/crossover.png)

#### e. Mutation:

*The objective of this function is to introduce randomness in the population by tweaking the weights in the Neural network (brain) of a creature.*

This function accepts a mutation rate as its parameter. The mutation rate, which is usually about 1 - 2%, is in fact the probability of introduction of randomness.

# Things to Improve

## 1. Sudden Muscle Movement

The muscles are controlled by the brain (*Neural Network*) of the creature. Each output of the neural network lies within the range of the sigmoid function, *i.e. [0, 1] and* maps to a certain muscle. Output value 1 indicates maximum muscle extension and 0 indicates maximum contraction. Due to this, subtle muscle movement is not possible. 

Consider, at time "*t",* one of the output nodes of the neural network is **1**. This will lead to full extension of the respective muscle (as an analogy, ***state 5** in the picture below*). Then at next instant "*t + 1*", the value of the same output node is **0 (*state 1*)**. Now, the currently fully flexed muscle will instantly get fully contracted (***from state 5 to state 1)***. This unnatural movement is not continuous and will exert immense force to the creature resulting in unwanted behavior. We want a continous movement that goes from state 5 to 4 to 3 and so on to state 1.

![](https://i.imgur.com/G5gcddL.jpg)

Due to this large spectrum of possible output states, the network takes a very long time to learn the movements. In fact the current topology ( *3 layer network* ) of the network might not even be sufficient to handle it.

### Possible Solution

- **Quantized Movement**: 
Allow only fixed possible extension of the muscle like in the picture above. This will reduce the learning time of the network. However, sudden movements might still be a problem.
- **Implement resting muscle length**:
Currently, the creatures don't have a fixed structure since their muscles don't have a resting length. There's no tendency of the muscles to stretch or contract to certain natural length; it is just a rigid connector.
By making the muscles "*spring-like"*, we can define the structure of the creatures. This is a better and more natural model of the muscles.

## 2. Sloppy Fitness Function

The first creature design successfully found a way to get to the right side of the screen despite its very sloppy and unnatural movement. This was because the creature was rewarded for the amount of distance it traveled towards right. There was no reward for staying up ( *keeping balance* ).

![](https://imgur.com/udPqUGm.gif) 
![](https://imgur.com/Khb27YD.gif)

This led to another problem. The creatures focused more on balancing than on walking.