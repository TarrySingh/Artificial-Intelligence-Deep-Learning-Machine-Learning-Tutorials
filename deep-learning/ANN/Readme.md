### Neural Networks
Neural Network - an acyclical directed graph. Neural nets have input layers, hidden layers and output layers. Layers are connected by weighted synapsis that multiply their input times the weight. Hidden layer consists of neurons that sum their inputs from synapsis and execute an activation function on the sum. The weights are intially set to random values but are trained with backpropagation.  The input and output are of fixed size. They are often called artificial neural networks, to distinguish it from biological nerons. Also called feedforward neural network to distinguish from more complicated neural nets with feedback mechanisms such as recurrent neural networks. 

<img src="https://github.com/andrewt3000/MachineLearning/blob/master/img/nn.png" height='250px' width='250px'/>

[neural nets](http://frnsys.com/ai_notes/machine_learning/neural_nets.html) thorough and concise study notes about neural networks.   

[Neural Networks demystified video](https://www.youtube.com/watch?v=bxe2T-V8XRs) - videos explaining neural networks. Includes [notes](https://github.com/stephencwelch/Neural-Networks-Demystified).    

[TensorFlow Neural Network Playground](http://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.28720&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false)  - This demo lets you run a neural network in your browser and see results graphically. Be sure to click the play button to start training. The network can easily train for the first three datasets with default parameters but the challenge is to get the network to train to the spiral dataset.  

[Deep learning on Udacity](https://classroom.udacity.com/courses/ud730/)  

### Features / Input
Features - measurable property being observed. In neural net context, it's  the input to a neural network.  Examples of features are pixel brightness in image object recognition, words encoded as vectors in nlp applications, audio signal in voice recognition applications.  
  
Feature selection - The process of choosing the features. It is important to pick features that correlate with the output. 

Dimensionality reduction - Reducing number of variables.  A simple example is selecting the area of a house as a feature rather than using width and length seperately. Other examples include singular value decomposition, auto-encoders, and t-SNE (for visualizations).      

Feature scaling - scale each feature to be in a common range typically -1 to 1 where 0 is the mean value.    

### Hyperparameters
Hyperparameters - the model’s parameters in a neural net such as architecture, learning rate, and regularization factor.	

Architecture - The structure of a neural network i.e. number of hidden layers, and number of nodes. 

Number of hidden layers - the higher the number of layers the more layers of abstraction it can represent. too many layers and the the network suffers from the vanishing or exploding gradient problem.  

Learning rate (&alpha;) - controls the size of the adjustments made during the training process. A typical value is .1 but often the value is a smaller number.  
if &alpha; is too low, convergance is slow.
if &alpha; is too high, there is no convergance, because it overshoots the local minimum.  
The learning rate is often reduced to a smaller number over time. This is often called annealing or decay. (examples: step decay, exponential decay)  

Underfitting - output doesn't fit the training data well.  
Overfitting - output fits training data well, but doesn't work well on test data.  

Regularization - a technique to minimize overfitting.  

L1 uses sum of absolute value of weights. L1 can yield sparse outputs.  
L2 uses sum of squared weights. L2 can't yield sparse outputs.    

[Dropout](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) - a form of regularization. "The key idea is to randomly drop units (along with their connections) from the neural network during training." Typical hyperparameter value is .5 (50%). As dropout value approaches zero, dropout has less effect, as it approaches 1 there are more connections are being zeroed out. The remaining active connections are scaled up to compensate for the zeroed out connections. See [Hinton's dropout in 3 lines of python](https://iamtrask.github.io/2015/07/28/dropout/) which features the following example:   

```python
layer_1 *= np.random.binomial([np.ones((len(X),hidden_dim))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))
```

### Activation Functions
Activation function - the "neuron" in the neural network executes an activation function on the sum of the weighted inputs. Typical activation functions include sigmoid, tanh, and ReLu.  

Sigmoid activation functions outputs a value between 0 and 1.  
```python
#sigmoid activation function using numpy
def sigmoid(z):
    return 1/(1+np.exp(-z))
```
Tanh activation function outputs value between -1 and 1.  

ReLu activation is typically the activation function used in state of the art convolutioanal neural nets for image processing. ReLu stands for rectified linear unit. It returns 0 for negative values, and the same number for positive values. for x < 0, y = 0. for x>0, y = x.  

```python
#relu activation function.
#numpy maximum - Compare two arrays and returns a new array containing the element-wise maxima
hidden_layer = np.maximum(0, np.dot(X, W) + b)
  
```


The softmax function is often used as the output's activation function and is useful for modeling probability distributions for multiclass classification where outputs are mutually exclusive (MNIST is an example). Output values are in range [0, 1]. The sum of outputs is 1. Use with cross entropy cost function.  
```python
def softmax(x):
    e2x = np.exp(x) 
    return e2x / np.sum(e2x, axis = 0)
```

### Training a network
Training data - Input and labeled output used as training examples. Data is typically split into training data, cross validation data and test data. Typical mix is 60% training, 20% validation and 20% testing data. Validation is used to tune the model and it's hyperparameters. Testing uses data that the model was never trained on.  

Training a network - minimize a cost function. Use backpropagation and gradient descent to adjust weights to make model more accurate. 

Steps to training a network.  
- initialize weights and biases.  
- training data is entered feedforward.  
- error is backpropagated.
- weights and biases are adjust based on learning rate.  

Number of times to iterate over the training data - You run the program until it hopefully converges on an acceptablely low error level. An epoch means the network has been been trained on every example once. You want to stop training if the validation data has an increasing error rate, this indicates overfitting. This is called early termination.   

Cost/error/objective/loss Function - measures how inaccurate a model is. Training a model minimizes the cost function. Sum of squared errors is a common cost function for regression. Cross entropy (aka log loss, negative log probability) is cost function for softmax function and multinomial logistic classification.   

Cross entropy function is sum of all the target values times the log of their output. Assuming the target output is 0 for all the wrong answers, and 1 for the correct answer, the correct answer is the only value that will contribute to the sum. The cost will be 0 if the correct output value is 1 because the log(1) is 0. The error approaches infinity as the output approaches 0 because the log of zero approaches infinity. See [Geoffery Hinton lecture](https://www.youtube.com/watch?v=mlaLLQofmR8)  

Backpropagation - Apply the chain rule to compute the gradients (partial derivative) of the loss function with respect to the weights in the network by moving backwards (output to input) through the network.  

#### Optimization algorithms
Batch gradient descent - Gradient descent is iteratively adjusting the weight by learning rate times the gradient to mimimize the error function. The term batch refers to the fact it uses the entire dataset. Batch works well for small datasets that have convex errors functions.  

Stochastic gradient descent is a variation of gradient descent that uses a single randomly choosen example to make an update to the weights. sgd is more scalable than batch graident descent and is used more often in practice for large scale deep learning. It's random nature makes it unlikely to get stuck in a local minima.  

Mini batch gradient descent: Stochastic gradient descent that considers more than one randomly choosen example befor making an update. Batch size is a hyperparmeter that determines how many training examples you consider before making a weight update. Typical values are factors of 2, such as 32 or 128.  

Momentum sgd is a variation that makes sgd less likely to go in the wrong direction because it collects data on each update in a velocity vector to assist in calculating the gradient. The velocity matrix represents the momentum. μ is a hyperparameter that represents the friction. μ is in the range of 0 to 1 and μ=1 is no friction.  

Other optimization algorithms include nesterov momentum sgd, adagrad, and adaDelta. See [An overview of gradient descent optimization algorithms](http://sebastianruder.com/optimizing-gradient-descent/)

[Practical tips for deep learning](http://yyue.blogspot.com/2015/01/a-brief-overview-of-deep-learning.html)  

### Other Types of Neural Networks
Convolutional Neural Networks - Specialized to process a grid of information such as an image. Convolution neural networks use filters (aka kernels) that convolve over the grid.    
[My notes on CNNs](https://github.com/andrewt3000/MachineLearning/blob/master/cnn4Images.md)

Recurrent Neural Network (RNN) - Used for input sequences such as text, audio, or video. RNNs are similar to a vanilla neural network but they also pass the hidden state as output of each neuron via a weighted connection as an input to the neurons in the same layer during the next sequence. RNNs are trained by backpropagation through time.  This feedback architecture allows the network to have memory of previous inputs. The memory is limited by vanishing/exploding gradient problem. Exploding gradient can be resolved by gradient clipping. A common hyperparameter is the number of steps to go back or "unroll" during training. There are variations such as bi-directional and recursive RNNs. 
[Code an RNN](https://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/)  

LSTM - [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf) - A specialized RNN that is capable of long term dependencies and mitigates the vanishing gradient problem. It contains memory cells and gate units. The number of memory cells is a hyperparameter. Memory cells pass memory information forward. The gates decide what information is stored in the memory cells. A vanilla LSTM has a forget gates, input gates and output gates. There are [many variations of the LSTM](http://arxiv.org/pdf/1503.04069.pdf).  
[Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) Blog post by Chris Olah.  
[RNN/LSTM tutorial for TensorFlow](https://www.tensorflow.org/versions/r0.10/tutorials/recurrent/index.html)  


GRU - Gated Recurrent Unit - Introduced by Cho. Another RNN variant similar but simpler than LSTM. It contains one update gate and combines the hidden state and memory cells among other differences.  

Beam Search - keep track of several of the most probable sequences and then choose the highest probability path. This lowers the risk a single bad probability in the sequence getting you on the wrong track. [See udacity video](https://classroom.udacity.com/courses/ud730/lessons/6378983156/concepts/63733319420923#)  
