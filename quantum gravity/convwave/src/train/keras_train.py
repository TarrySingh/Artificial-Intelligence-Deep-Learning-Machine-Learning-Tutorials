# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import os
import numpy as np
import h5py
import datetime
import sys

from sklearn.model_selection import train_test_split
from scipy.spatial.distance import hamming
from scipy import signal
from functools import partial

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

import keras.backend as K
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Reshape, BatchNormalization, \
    Activation, Dropout
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, Callback, TensorBoard
from keras.losses import binary_crossentropy


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def whiten(image):
    mittelwert = 18.4455575155
    standardabweichung = 3.66447104786
    return (image - mittelwert) / standardabweichung


def hamming_dist(y_true, y_pred):
    return K.mean(np.abs(y_pred - y_true))


def fuzzy_binary_crossentropy(weights):

    def loss(y_true, y_pred):

        y_true = K.minimum(y_true, weights)
        y_pred = K.minimum(y_pred, weights)
        return binary_crossentropy(y_true, y_pred)

    return loss


def make_grayzones(label, size=10):
    window = signal.hann(size)
    filtered = signal.convolve(label, window, mode='same') / sum(window)
    grayzone = np.fromiter(map(lambda x: 1-int(0 < x < 1), filtered),
                           dtype=np.int)
    return grayzone


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class PrintLearningRate(Callback):

    def on_epoch_begin(self, epoch, logs=None):
            lr = K.eval(self.model.optimizer.lr)
            print('\nLearning Rate:', lr, end='\n', flush=True)


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    data_path = '../data/'
    print('Starting main routine...')

    # -------------------------------------------------------------------------
    # LOAD DATA AND SPLIT TRAINING AND TEST SAMPLE
    # -------------------------------------------------------------------------

    filename = os.path.join(data_path, 'training_samples_400_800.h5')

    with h5py.File(filename, 'r') as file:

        x = np.array(file['spectrograms'])
        y = np.array(file['labels'])

    # Reshape to make it work with keras
    y = y.reshape((y.shape[0], y.shape[1], 1)).astype('int')

    print('x:', x.shape)
    print('y:', y.shape)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,
                                                        random_state=42)

    # -------------------------------------------------------------------------
    # CALCULATE WEIGHTS AND PREPARE CUSTOM LOSS FUNCTION
    # -------------------------------------------------------------------------

    weights = np.array([make_grayzones(_.squeeze()) for _ in y_train])
    # weights = weights.reshape(y_train.shape)
    # custom_loss = partial(fuzzy_binary_crossentropy,
    #                      weights=K.variable(weights))
    print(weights.shape)
    custom_loss = fuzzy_binary_crossentropy(weights)

    # -------------------------------------------------------------------------
    # DEFINE THE MODEL
    # -------------------------------------------------------------------------

    print('Defining the model...')

    model = Sequential()
    # -------------------------------------------------------------------------
    model.add(Conv2D(128, (3, 7),
                     input_shape=x_train[0].shape,
                     data_format='channels_last',
                     padding='same',
                     kernel_initializer='random_uniform',
                     name='Start'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    # -------------------------------------------------------------------------
    for i in range(6):
        model.add(Conv2D(128, (3, 7),
                         padding='same',
                         kernel_initializer='random_uniform',
                         name=str(i)))
        model.add(BatchNormalization())
        model.add(Activation('elu'))
        model.add(MaxPooling2D(pool_size=(2, 1)))
        model.add(Dropout(rate=0.4))
    # -------------------------------------------------------------------------
    model.add(Conv2D(1, (1, 1),
                     activation='sigmoid',
                     kernel_initializer='random_uniform',
                     name="Ende"))
    # -------------------------------------------------------------------------
    model.add(Reshape((-1, 1),
                      name="Reshape"))

    optimizer = Adam(lr=0.001)
    model.compile(loss=binary_crossentropy,
                  sample_weight_mode='temporal',
                  optimizer=optimizer,
                  metrics=[hamming_dist])

    # -------------------------------------------------------------------------
    # DEFINE CALLBACKS FOR THE TRAINING
    # -------------------------------------------------------------------------

    # Callback to print the current learning rate at every epoch
    print_learning_rate = PrintLearningRate()

    # Callback to reduce the learning rate if we are entering a plateau
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.707, patience=8,
                                  epsilon=1e-03, min_lr=1e-08)

    # Callback to create unique TensorBoard log directories
    tensorboard = TensorBoard(log_dir='./logs/{:%Y-%m-%d_%H:%M:%S}'.
                              format(datetime.datetime.now()))

    # -------------------------------------------------------------------------
    # FIT THE MODEL AND SAVE THE WEIGHTS
    # -------------------------------------------------------------------------

    model.load_weights('model_weights.h5', by_name=False)
    model.fit(x_train, y_train,
              batch_size=16,
              epochs=100,
              validation_split=0.1,
              sample_weight=weights,
              callbacks=[reduce_lr, print_learning_rate, tensorboard])
    model.save_weights('model_weights_faintest.h5')

    # -------------------------------------------------------------------------
    # MAKE PREDICTIONS AND EVALUATE THE ACCURACY
    # -------------------------------------------------------------------------

    # np.set_printoptions(threshold=np.inf)

    # Make predictions on the test set
    y_pred = np.round(model.predict(x_test))

    n_correct = 0
    hamming_distances = []

    # Loop over the test set and count the correct predictions and calculate
    # the Hamming distances between truth and prediction
    for i in range(len(y_test)):
        if all(y_pred[i] == y_test[i]):
            n_correct += 1
        hamming_distances.append(hamming(y_pred[i], y_test[i]))

    # Calculate the average Hamming distance between prediction and true label
    average_hamming_distance = np.mean(hamming_distances)

    print()
    print('Fraction of Correct Ones: {:.3f}'.format(n_correct / len(y_test)))
    print('Average Hamming Distance: {:.3f}'.format(average_hamming_distance))

    # Save test set and predictions in another HDF file (for manual inspection)
    filename = os.path.join(data_path, 'test_predictions.h5')
    with h5py.File(filename, 'w') as file:

        file['x'] = np.array(x_test)
        file['y_pred'] = np.array(y_pred)
        file['y_true'] = np.array(y_test)
