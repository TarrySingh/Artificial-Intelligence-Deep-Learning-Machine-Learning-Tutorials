# Importing TF, Mnist
from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('tmp/data/', one_hot=True)

# Define hidden layers
n_nodes_hlayer1 = 500
n_nodes_hlayer2 = 500
n_nodes_hlayer3 = 500

# define how many runs and how many pics to run to do backprop, manipulate wgts and repeat & no of epochs you want to run
n_classes = 10
batch_size = 100
hm_epoch = 10
logs_path = '/tmp/tensorflow_logs/example'

# Height & Width / Input features of size = 28x28 pix = 784

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def nn_model(data):
    # simply put this model = inputs data*weights + biases
    hidden_layer_1 = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hlayer1])),
        'biases':tf.Variable(tf.random_normal([n_nodes_hlayer1]))}

    hidden_layer_2 = {'weights':tf.Variable(tf.random_normal([n_nodes_hlayer1, n_nodes_hlayer2])),
        'biases':tf.Variable(tf.random_normal([n_nodes_hlayer2]))}

    hidden_layer_3 = {'weights':tf.Variable(tf.random_normal([n_nodes_hlayer2, n_nodes_hlayer3])),
    'biases':tf.Variable(tf.random_normal([n_nodes_hlayer3]))}
    
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hlayer3, n_classes])),
        'biases':tf.Variable(tf.random_normal([n_classes]))}

    layer1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
    layer1 = tf.nn.relu(layer1)

    layer2 = tf.add(tf.matmul(layer1, hidden_layer_2['weights']), hidden_layer_2['biases'])
    layer2 = tf.nn.relu(layer2)
    
    layer3 = tf.add(tf.matmul(layer2, hidden_layer_3['weights']), hidden_layer_3['biases'])
    layer3 = tf.nn.relu(layer3)
    
    output = tf.add(tf.matmul(layer3, output_layer['weights']), output_layer['biases'])
    
    return output

# We define function to train this NN
def train_nn(x):
    prediction = nn_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    # Watch out to pick Adam which is same as SGD
    optimizer = tf.train.AdamOptimizer().minimize(cost) #WTF Adadelta!!!!
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) #This changed in TF 1.0 version onwards
        # training it
        for epoch in range (hm_epoch):
            epoch_loss = 0
            for _ in range ( int ( mnist.train.num_examples / batch_size)):
                epoch_x , epoch_y = mnist.train.next_batch (batch_size)
                _ , c = sess.run([optimizer, cost], feed_dict={x:epoch_x, y:epoch_y})
                #code that will optimize the weights and biases
                epoch_loss += c
            print('Epochs', epoch, 'completed out of', hm_epoch, 'loss:', epoch_loss)
        
        #Testing
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast (correct, 'float'))
        print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_nn(x)

# Examine results with Tensorboard

print("Run the command line:\n"
      "--> tensorboard --logdir=/tmp/tensorflow_logs "
      "\nThen open http://0.0.0.0:6006/ into your web browser")
