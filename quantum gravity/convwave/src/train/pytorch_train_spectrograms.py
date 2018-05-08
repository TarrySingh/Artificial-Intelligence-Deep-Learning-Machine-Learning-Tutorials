# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import h5py
import os
import sys
import time
import datetime
import warnings

from tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import lr_scheduler

from models import SpectrogramFCN
from IPython import embed


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def create_weights(label, start_size=40, end_size=3):
    """
    Create the weights ('grayzones') for a given label.

    Args:
        label: A vector labeling the pixels that contain an injection
        start_size: Number of pixels to ignore at the start of an injection
        end_size: Number of pixels to ignore at the end of an injections

    Returns: A vector that is 0 for the pixels that should be ignored and 1
        for all other pixels.
    """

    a = np.logical_xor(label, np.roll(label, 1))
    b = np.cumsum(a) % 2

    if start_size == 0:
        c = np.zeros(label.shape)
    else:
        c = np.convolve(a * b, np.hstack((np.zeros(start_size - 1),
                                          np.ones(start_size))),
                        mode="same")

    if end_size == 0:
        d = np.zeros(label.shape)
    else:
        d = np.convolve(a * np.logical_not(b),
                        np.hstack((np.ones(end_size), np.zeros(end_size - 1))),
                        mode="same")

    return np.logical_not(np.logical_or(c, d)).astype('int')


def hamming_dist(y_true, y_pred):
    """
    Calculate the Hamming distance between a given predicted label and the
    true label.

    Args:
        y_true: The true label
        y_pred: The predicted label

    Returns: The Hamming distance between the two vectors
    """

    return np.mean(np.abs(y_true - y_pred), axis=(1, 0))


def progress_bar(current_value, max_value, start_time, **kwargs):
    """
    Print the progress bar during training that contains all relevant
    information, i.e. number of epochs, percentage of processed mini-batches,
    elapsed time, estimated time remaining, as well as all metrics provided.

    Args:
        current_value: Current number of processed mini-batches
        max_value: Number of total mini-batches
        start_time: Absolute timestamp of the moment the epoch began
        **kwargs: Various metrics, e.g. the loss or Hamming distance
    """

    # Some preliminary definitions
    bar_length = 20
    elapsed_time = time.time() - start_time

    # Construct the actual progress bar
    percent = float(current_value) / max_value
    bar = '=' * int(round(percent * bar_length))
    spaces = '-' * (bar_length - len(bar))

    # Calculate the estimated time remaining
    eta = elapsed_time / percent - elapsed_time

    # Start with the default info: Progress Bar, number of processed
    # mini-batches, time elapsed, and estimated time remaining (the '\r' at
    # the start moves the carriage back the start of the line, meaning that
    # the progress bar will be overwritten / updated!)
    out = ("\r[{0}] {1:>3}% ({2:>2}/{3}) | {4:.1f}s elapsed | "
           "ETA: {5:.1f}s | ".format(bar + spaces, int(round(percent * 100)),
                                     int(current_value), int(max_value),
                                     elapsed_time, eta))

    # Add all provided metrics, e.g. loss and Hamming distance
    metrics = []
    for metric, value in sorted(kwargs.items()):
        if metric != 'lr':
            metrics.append("{}: {:.3f}".format(metric, value))
        else:
            metrics.append("{}: {:.8f}".format(metric, value))
    out += ' - '.join(metrics) + ' '

    # Actually write the finished progress bar to the command line
    sys.stdout.write(out)
    sys.stdout.flush()


def load_data_as_tensor_datasets(file_path, split_ratios=(0.7, 0.2, 0.1),
                                 shuffle_data=False, random_seed=42):
    """
    Take an HDF file with data (Gaussian Noise with waveform injections) and
    read it in, split it into training, test and validation data, and convert
    it to PyTorch TensorDatasets, which can be used in PyTorch DataLoaders,
    which are in turn useful for looping over the data in mini-batches.

    Args:
        file_path: The path to the HDF file containing the samples.
        split_ratios: The ratio of training:test:validation. This ought to
            sum up to 1!
        shuffle_data: Whether or not to shuffle the data before splitting.
        random_seed: Seed for the random number generator.

    Returns: Spectrograms and their respective labels, combined in a PyTorch
        TensorDataset, for training, test and validation.
    """

    # TODO: We might also want to pre-process (normalize) the data?

    # Set the seed for the random number generator
    np.random.seed(random_seed)

    # Read in the spectrograms from the HDF file
    with h5py.File(file_path, 'r') as file:

        x = np.array(file['spectrograms'])
        y = np.array(file['labels'])

    # Swap axes around to get to NCHW format
    x = np.swapaxes(x, 1, 3)
    x = np.swapaxes(x, 2, 3)
    x = np.squeeze(x)

    # Generate the indices for training, test and validation
    idx = np.arange(len(x))

    # Shuffle the indices (data) if requested
    if shuffle_data:
        idx = np.random.permutation(idx)

    # Get the indices for training, test and validation
    splits = np.cumsum(split_ratios)
    idx_train = idx[:int(splits[0]*len(x))]
    idx_test = idx[int(splits[0]*len(x)):int(splits[1]*len(x))]
    idx_validation = idx[int(splits[1]*len(x)):]

    # Select the actual data using these indices
    x_train, y_train = x[idx_train], y[idx_train]
    x_test, y_test = x[idx_test], y[idx_test]
    x_validation, y_validation = x[idx_validation], y[idx_validation]

    # Convert the training and test data to PyTorch / CUDA tensors
    x_train = torch.from_numpy(x_train).float().cuda()
    y_train = torch.from_numpy(y_train).float().cuda()
    x_test = torch.from_numpy(x_test).float().cuda()
    y_test = torch.from_numpy(y_test).float().cuda()
    x_validation = torch.from_numpy(x_validation).float().cuda()
    y_validation = torch.from_numpy(y_validation).float().cuda()

    # Create TensorDatasets for training, test and validation
    tensor_dataset_train = TensorDataset(x_train, y_train)
    tensor_dataset_test = TensorDataset(x_test, y_test)
    tensor_dataset_validation = TensorDataset(x_validation, y_validation)

    # Return the resulting TensorDatasets
    return tensor_dataset_train, tensor_dataset_test, tensor_dataset_validation


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    warnings.warn('This script is not up to date! Do not trust anything that '
                  'comes out of here! Just stick to TimeSeries for now :)')

    print('Starting main routine...')

    #
    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    # Which distances and sample size are we using?
    distances = '0100_0300'
    sample_size = '256'

    # Where does our data live and which file should we use?
    data_path = '../data/'
    file_name = 'samples_spectrograms_{}_{}.h5'.format(distances, sample_size)

    file_path = os.path.join(data_path, 'training', 'spectrograms', file_name)

    #
    # -------------------------------------------------------------------------
    # LOAD DATA, SPLIT TRAINING AND TEST SAMPLE, AND CREATE DATALOADERS
    # -------------------------------------------------------------------------

    print('Reading in data...', end=' ')

    # Load the data from the HDF file, split it, and convert to TensorDatasets
    tensor_datasets = load_data_as_tensor_datasets(file_path)
    data_train, data_test, data_validation = tensor_datasets

    print('Done!')

    # -------------------------------------------------------------------------
    # SET UP THE NET
    # -------------------------------------------------------------------------

    # Set up the net and make it CUDA ready; activate GPU parallelization
    model = SpectrogramFCN()
    model.float().cuda()
    model = torch.nn.DataParallel(model)

    # If desired, load weights from pre-trained model
    # TODO: Make this accessible through command line options!
    # weights_file = './weights/spectrogram_weights_0100_0300_4k.net'
    # net.load_state_dict(torch.load(weights_file))

    # Set up the optimizer and the initial learning rate, and zero parameters
    initial_lr = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    optimizer.zero_grad()

    # Set up the learning schedule to reduce the LR on plateaus
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                               factor=0.5, patience=5,
                                               threshold=0.01, )

    # Set the mini-batch size, and calculate the number of mini-batches
    batch_size = 16
    n_minibatches_train = np.ceil(len(data_train) / batch_size)
    n_minibatches_test = np.ceil(len(data_test) / batch_size)
    n_minibatches_validation = np.ceil(len(data_validation) / batch_size)

    # Fix the number of epochs to train for
    n_epochs = 10

    threshold = 0.01 * 10 ** (-21)

    # -------------------------------------------------------------------------
    # SET UP A LOGGER FOR TENSORBOARD VISUALIZATION
    # -------------------------------------------------------------------------

    run_start = datetime.datetime.now()
    log_name = [run_start, distances, sample_size, initial_lr, threshold]
    log_name_formatted = '[{:%Y-%m-%d_%H:%M}]-[{}]-[{}]-[lr_{:.1e}]-'\
                         '[thresh_{:.2e}]'.format(*log_name)
    writer = SummaryWriter(log_dir='logs/{}'.format(log_name_formatted))
    writer.add_text(tag='Description',
                    text_string='(Description missing.)')

    # -------------------------------------------------------------------------
    # TRAIN THE NET FOR THE GIVEN NUMBER OF EPOCHS
    # -------------------------------------------------------------------------

    print('\nStart training: Training on {} examples, validating on {} '
          'examples\n'.format(len(data_train), len(data_validation)))

    # -------------------------------------------------------------------------

    for epoch in range(n_epochs):

        print('Epoch {}/{}'.format(epoch+1, n_epochs))

        running_loss = 0
        running_hamm = 0
        start_time = time.time()

        #
        # ---------------------------------------------------------------------
        # LOOP OVER MINI-BATCHES AND TRAIN THE NETWORK
        # ---------------------------------------------------------------------

        for mb_idx, mb_data in enumerate(DataLoader(data_train,
                                                    batch_size=batch_size)):

            # Get the inputs and wrap them in a PyTorch variable
            inputs, labels = mb_data
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

            # Get the size of the mini-batch
            mb_size = len(labels)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass through the net and reshape outputs properly
            outputs = model.forward(inputs)
            outputs = outputs.view((outputs.size()[0], outputs.size()[-1]))

            # Calculate weights and set up the loss function
            weights = torch.eq(torch.gt(labels, 0) *
                               torch.lt(labels, threshold), 0).float().cuda()

            loss_function = nn.BCELoss(weight=weights.float().cuda(),
                                       size_average=True).cuda()

            # Calculate the loss
            loss = loss_function(outputs.cuda(), torch.ceil(labels).cuda())
            running_loss += float(loss.data.cpu().numpy())

            # Use back-propagation to update the weights according to the loss
            loss.backward()
            optimizer.step()

            # Calculate the hamming distance between prediction and truth
            weighted_pred = (weights.float() * outputs).data.cpu().numpy()
            weighted_true = (weights.float() * labels).data.cpu().numpy()
            running_hamm += hamming_dist(np.round(weighted_pred),
                                         weighted_true)

            # Make output to the command line
            progress_bar(current_value=mb_idx+1,
                         max_value=n_minibatches_train,
                         start_time=start_time,
                         loss=running_loss/(mb_idx+1),
                         hamming_dist=running_hamm/(mb_idx+1))

        #
        # ---------------------------------------------------------------------
        # LOOP OVER MINI-BATCHES AND EVALUATE ON VALIDATION SAMPLE
        # ---------------------------------------------------------------------

        # At the end of an epoch, calculate the validation loss
        val_loss = 0
        val_hamm = 0

        # Process validation data in mini-batches
        for mb_idx, mb_data in enumerate(DataLoader(data_validation,
                                                    batch_size=batch_size)):
            # Calculate the loss for a particular mini-batch
            inputs, labels = mb_data
            inputs = Variable(inputs, volatile=True).cuda()
            labels = Variable(labels, volatile=True).cuda()

            # Forward pass through the net and reshape outputs properly
            outputs = model.forward(inputs)
            outputs = outputs.view((outputs.size()[0], outputs.size()[-1]))

            # Get the size of the mini-batch
            mb_size = len(labels)

            # Calculate weights and set up the loss function
            weights = torch.eq(torch.gt(labels, 0) *
                               torch.lt(labels, threshold), 0).float().cuda()

            loss_function = nn.BCELoss(weight=weights.float().cuda(),
                                       size_average=True).cuda()

            # Calculate the loss
            loss = loss_function(outputs.cuda(), torch.ceil(labels).cuda())
            val_loss += float(loss.data.cpu().numpy())

            # Calculate the hamming distance between prediction and truth
            weighted_pred = (weights.float() * outputs).data.cpu().numpy()
            weighted_true = (weights.float() * labels).data.cpu().numpy()
            val_hamm += hamming_dist(np.round(weighted_pred), weighted_true)

        #
        # ---------------------------------------------------------------------
        # PRINT FINAL PROGRESS BAR AND LOG STUFF FOR TENSORBOARD VISUALIZATION
        # ---------------------------------------------------------------------

        # Get the current learning rate... TODO: is this really the only way?!
        lr = None
        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        # Plot the final progress bar for this epoch
        progress_bar(current_value=n_minibatches_train,
                     max_value=n_minibatches_train,
                     start_time=start_time,
                     loss=running_loss/n_minibatches_train,
                     hamming_dist=running_hamm/n_minibatches_train,
                     val_loss=val_loss/n_minibatches_validation,
                     val_hamming_dist=val_hamm/n_minibatches_validation,
                     lr=lr)
        print()

        # Save everything to the TensorBoard logger
        def log(name, value):
            writer.add_scalar(name, value, epoch)

        log('loss', running_loss/n_minibatches_train)
        log('hamming_dist', running_hamm/n_minibatches_train)
        log('val_loss', val_loss/n_minibatches_validation)
        log('val_hamming_dist', val_hamm/n_minibatches_validation)
        log('learning_rate', lr)

        #
        # ---------------------------------------------------------------------
        # SAVE SNAPSHOTS OF THE MODEL'S WEIGHTS (EVERY N EPOCHS)
        # ---------------------------------------------------------------------

        if epoch % 1 == 0:

            # Check if the appropriate directory for this run exists
            snapshot_dir = os.path.join('./weights/', log_name_formatted)
            if not os.path.exists(snapshot_dir):
                os.makedirs(snapshot_dir)

            # Save the weights for the current epoch ("snapshot")
            dummy = [distances, sample_size, epoch]
            weights_file_name = 'weights_{}_{}_epoch-{:03d}.net'.format(*dummy)
            weights_file_path = os.path.join(snapshot_dir, weights_file_name)
            torch.save(model.state_dict(), weights_file_path)

            # TODO: Maybe delete the older snapshots?

        #
        # ---------------------------------------------------------------------
        # REDUCE THE LEARNING RATE IF APPROPRIATE
        # ---------------------------------------------------------------------

        scheduler.step(val_loss/n_minibatches_validation)

    # -------------------------------------------------------------------------

    print('Finished Training!')
    writer.close()

    # Save the trained model
    print('Saving model...', end=' ')
    weights_file = ('./weights/spectrograms_weights_{}_{}.net'.
                    format(distances, sample_size))
    torch.save(model.state_dict(), weights_file)
    print('Done!')

    #
    # -------------------------------------------------------------------------
    # MAKE PREDICTIONS ON THE TEST SET
    # -------------------------------------------------------------------------

    print('Start making predictions on the test sample...', end=' ')

    # Convert test data to numpy arrays that can be stored in an HDF file
    x_test = data_test.data_tensor.cpu().numpy()
    y_test = data_test.target_tensor.cpu().numpy()

    # Initialize an empty array for our predictions
    y_pred = []

    # Loop over the test set (in mini-batches) to get the predictions
    for mb_idx, mb_data in enumerate(DataLoader(data_test,
                                                batch_size=batch_size)):

        # Calculate the loss for a particular mini-batch
        inputs, labels = mb_data
        inputs = Variable(inputs, volatile=True).cuda()
        labels = Variable(labels, volatile=True).cuda()

        # Make predictions for the given mini-batch
        outputs = model.forward(inputs)
        outputs = outputs.view((outputs.size()[0], outputs.size()[-1]))
        outputs = outputs.data.cpu().numpy()

        # Stack that onto the previous predictions
        y_pred.append(outputs)

    y_pred = np.vstack(y_pred)

    # Set up the name and directory of the file where the predictions will
    # be saved.
    test_predictions_file = 'predictions_spectrograms_{}_{}.h5'.\
        format(distances, sample_size)
    test_predictions_path = os.path.join(data_path, 'predictions',
                                         test_predictions_file)

    with h5py.File(test_predictions_path, 'w') as file:

        file['x'] = x_test
        file['y_pred'] = y_pred
        file['y_true'] = y_test

    print('Done!')
