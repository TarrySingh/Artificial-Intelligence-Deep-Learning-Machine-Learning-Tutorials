# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import h5py
import os
import time
import datetime
import pprint

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from models import TimeSeriesFCN
from training_tools import load_data_as_tensor_datasets, progress_bar, \
    hamming_dist, apply_model, get_current_lr, get_weights, get_labels, \
    TrainingArgumentParser

from IPython import embed


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # CHECK IF CUDA IS AVAILABLE
    # -------------------------------------------------------------------------

    if torch.cuda.is_available():
        print('CUDA is available, will use GPUs for training!')
    else:
        print('CUDA is not available, will use CPUs for training!')

    # -------------------------------------------------------------------------
    # PARSE COMMAND LINE ARGUMENTS AND DEFINE GLOBAL PARAMETERS
    # -------------------------------------------------------------------------

    # Parse command line arguments
    parser = TrainingArgumentParser()
    arguments = parser.parse_args()

    # Print arguments that will be used
    print('Beginning training with following parameters:')
    pprint.pprint(arguments)

    # Define shortcuts for arguments
    batch_size = arguments['batch_size']
    description = arguments['description']
    distances = arguments['distances']
    initial_lr = arguments['initial_lr']
    n_epochs = arguments['n_epochs']
    noise_source = arguments['noise_source']
    regularization_parameter = arguments['regularization_parameter']
    sample_size = arguments['sample_size']
    threshold = arguments['threshold']
    weights_file_name = arguments['weights_file_name']

    # -------------------------------------------------------------------------
    # BUILD PATHS FOR THE FILE WE WILL BE USING
    # -------------------------------------------------------------------------

    print('Building file paths...', end=' ')

    # Base path of data directory
    data_path = '../data/'

    # Build the path for the file where our training samples come from
    sample_file_name = 'training_{}_{}_{}_FWHM.h5'.format(noise_source,
                                                          distances,
                                                          sample_size)
    sample_file_path = os.path.join(data_path, 'training', 'timeseries',
                                    sample_file_name)

    # Build the path for the weights file we want to load
    if weights_file_name is not None:
        weight_file_path = os.path.join('.', 'weights', weights_file_name)
    else:
        weight_file_path = None

    # Build path for the HDF file in which we will store the test predictions
    pred_file_name = 'training_predictions_{}_{}_{}_FWHM.h5'.\
        format(noise_source, distances, sample_size)
    pred_file_path = os.path.join(data_path, 'predictions', 'timeseries',
                                  'training', pred_file_name)

    print('Done!')

    # -------------------------------------------------------------------------
    # LOAD DATA, SPLIT TRAINING AND TEST SAMPLE, AND CREATE DATALOADERS
    # -------------------------------------------------------------------------

    print('Reading in data...', end=' ')

    # Load the data from the HDF file, split it, and convert to TensorDatasets
    tensor_datasets = load_data_as_tensor_datasets(sample_file_path)
    data_train, data_test, data_validation = tensor_datasets

    print('Done!')

    # -------------------------------------------------------------------------
    # INITIALIZE THE MODEL AND LOAD WEIGHTS IF NECESSARY
    # -------------------------------------------------------------------------

    # Define the model
    model = TimeSeriesFCN()
    model = model.float()

    # If CUDA is available, use it and activate GPU parallelization
    if torch.cuda.is_available():
        model.float().cuda()
        model = torch.nn.DataParallel(model)

    # If desired, load weights file to warm-start the training
    if weight_file_path is not None:
        print('Loading weights for warm start...', end=' ')
        model.load_state_dict(torch.load(weight_file_path))
        print('Done!')

    # -------------------------------------------------------------------------
    # SET UP THE OPTIMIZER USED FOR TRAINING THE NET
    # -------------------------------------------------------------------------

    # Don't give parameters to the optimizer that don't need a gradient
    params_for_opt = filter(lambda p: p.requires_grad, model.parameters())

    # Set up the optimizer and make sure all gradients are zero
    optimizer = optim.Adam(params_for_opt, lr=initial_lr)
    optimizer.zero_grad()

    # -------------------------------------------------------------------------
    # SET UP THE SCHEDULER THAT REDUCES THE LEARNING RATE ON PLATEAUS
    # -------------------------------------------------------------------------

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                               factor=0.707, patience=3,
                                               threshold=0.01)

    # -------------------------------------------------------------------------
    # DEFINE THE LOSS FUNCTION THAT WE WILL BE OPTIMIZING FOR
    # -------------------------------------------------------------------------

    def loss_function(y_pred, y_true, weights, reg=regularization_parameter):
        """
        Calculate the loss as the weighted sum of a Binary Cross-Entropy
        term and a Total Variation penalty term.

        Args:
            y_pred: The tensor of predictions
            y_true: The tensor of true labels
            weights: The weights that encode the fuzzy zones
            reg: The regularization parameter

        Returns:
            loss: The loss value for y_pred and y_true
        """

        # Set up the Binary Cross-Entropy term of the loss
        bce_loss = nn.BCELoss(weight=weights)
        if torch.cuda.is_available():
            bce_loss = bce_loss.cuda()

        # Set up the Total Variation term of the loss. Only calculate it if
        # we are actually going to use it though!
        if reg > 0:
            tv_loss = torch.sum(torch.abs(weights[:, :-1] * y_pred[:, :-1] -
                                          weights[:, 1:] * y_pred[:, 1:]))
        else:
            tv_loss = 0

        # Return the weighted sum of BCE loss and TV loss
        return bce_loss(y_pred, y_true) + reg * tv_loss

    # -------------------------------------------------------------------------
    # SET UP REMAINING PARAMETERS
    # -------------------------------------------------------------------------

    # Calculate the number of mini-batches
    n_minibatches_train = np.ceil(len(data_train) / batch_size)
    n_minibatches_test = np.ceil(len(data_test) / batch_size)
    n_minibatches_validation = np.ceil(len(data_validation) / batch_size)

    # Keep track of all the metrics we want to log
    metrics = {'loss': [], 'hamming': [], 'val_loss': [], 'val_hamming': []}

    # -------------------------------------------------------------------------
    # SET UP A LOGGER FOR TENSORBOARD VISUALIZATION
    # -------------------------------------------------------------------------

    # Define a log directory and set up a writer
    run_start = datetime.datetime.now()
    log_name = [run_start, noise_source, distances, sample_size, initial_lr,
                threshold, regularization_parameter]
    log_name_formatted = '[{:%Y-%m-%d_%H:%M}]-[{}]-[{}]-[{}]-[LR_{:.1e}]-'\
                         '[THR_{:.1f}]_[REG_{:.2e}]'.format(*log_name)
    writer = SummaryWriter(log_dir='logs/{}'.format(log_name_formatted))
    writer.add_text(tag='Description', text_string=description)

    # Define a shortcut to write metrics to the log
    def log_metric(name, value, epoch):
        writer.add_scalar(name, value, epoch)

    # -------------------------------------------------------------------------
    # TRAIN THE NET FOR THE GIVEN NUMBER OF EPOCHS
    # -------------------------------------------------------------------------

    print('\nStart training: Training on {} examples, validating on {} '
          'examples\n'.format(len(data_train), len(data_validation)))

    # -------------------------------------------------------------------------
    #

    for epoch in range(n_epochs):

        # Print the current epoch of the training
        print('Epoch {}/{}'.format(epoch+1, n_epochs))

        # Keep logging the losses and hamming distances of all mini-batches
        epoch_losses = []
        epoch_hammings = []

        # Start the stopwatch for this epoch to get an ETA
        start_time = time.time()

        # ---------------------------------------------------------------------
        # LOOP OVER MINI-BATCHES AND TRAIN THE NETWORK
        # ---------------------------------------------------------------------

        # Reset data loader for this epoch to shuffle training data
        data_loader_train = DataLoader(data_train,
                                       batch_size=batch_size,
                                       shuffle=True)

        # Loop in mini-batches over the training data
        for mb_idx, mb_data in enumerate(data_loader_train):

            # Get the inputs and wrap them in a PyTorch variable
            inputs, raw_labels = mb_data
            inputs, raw_labels = Variable(inputs), Variable(raw_labels)

            # If CUDA is available, run everything on the GPU
            if torch.cuda.is_available():
                inputs, raw_labels = inputs.cuda(), raw_labels.cuda()

            # Calculate the real labels from the raw labels
            labels = get_labels(raw_labels, threshold)

            # Run a forward pass through the net and reshape outputs properly
            outputs = model.forward(inputs)
            outputs = outputs.view((outputs.size()[0], outputs.size()[-1]))

            # Calculate weights (i.e., fuzzy zones) from labels
            weights = get_weights(raw_labels, threshold)

            # Calculate the loss using the weighted labels and predictions
            # and keep track of it for logging purposes
            loss = loss_function(y_pred=outputs,
                                 y_true=labels,
                                 weights=weights)
            mb_loss = float(loss.data.cpu().numpy())
            epoch_losses.append(mb_loss)

            # Calculate the Hamming distance for this mini-batch
            mb_hamming = hamming_dist(y_pred=(outputs * weights),
                                      y_true=(labels * weights))
            epoch_hammings.append(mb_hamming)

            # Zero the gradients, then back-propagate the loss and update the
            # weights of the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Make output to the command line
            progress_bar(current_value=mb_idx+1,
                         max_value=n_minibatches_train,
                         start_time=start_time,
                         loss=np.mean(epoch_losses),
                         hamming=np.mean(epoch_hammings))

        # ---------------------------------------------------------------------
        # EVALUATE LOSS AND HAMMING DISTANCE ON VALIDATION SAMPLE
        # ---------------------------------------------------------------------

        # Set up the data load for the validation data
        data_loader_validation = DataLoader(data_validation,
                                            batch_size=batch_size)

        # Apply the net to the validation data and get the outputs
        outputs = apply_model(model, data_loader_validation)

        # Get the true labels for the validation data and calculate weights
        raw_labels = Variable(data_validation.target_tensor, volatile=True)
        labels = get_labels(raw_labels, threshold)
        weights = get_weights(raw_labels, threshold)

        # Calculate the validation loss
        val_loss = loss_function(y_pred=outputs,
                                 y_true=labels,
                                 weights=weights)
        val_loss = float(val_loss.data.cpu().numpy())

        # Calculate the validation Hamming distance
        val_hamming = hamming_dist(y_pred=(outputs * weights),
                                   y_true=(labels * weights))

        # ---------------------------------------------------------------------
        # STORE ALL METRICS FOR THIS EPOCH
        # ---------------------------------------------------------------------

        metrics['loss'].append(np.mean(epoch_losses))
        metrics['hamming'].append(np.mean(epoch_hammings))
        metrics['val_loss'].append(val_loss)
        metrics['val_hamming'].append(val_hamming)

        # ---------------------------------------------------------------------
        # PRINT FINAL PROGRESS BAR FOR EPOCH
        # ---------------------------------------------------------------------

        # Get the current learning rate from the optimizer
        lr = get_current_lr(optimizer)

        # Plot the final progress bar for this epoch
        progress_bar(current_value=n_minibatches_train,
                     max_value=n_minibatches_train,
                     start_time=start_time,
                     loss=metrics['loss'][epoch],
                     hamming=metrics['hamming'][epoch],
                     val_loss=metrics['val_loss'][epoch],
                     val_hamming=metrics['val_hamming'][epoch],
                     lr=lr,
                     thresh='{:.2e}'.format(threshold),
                     end='\n')

        # ---------------------------------------------------------------------
        # LOG EPOCH METRICS TO THE TENSORBOARD LOGGER
        # ---------------------------------------------------------------------

        log_metric('loss', np.mean(epoch_losses), epoch)
        log_metric('hamming_dist', np.mean(epoch_hammings), epoch)
        log_metric('val_loss', val_loss, epoch)
        log_metric('val_hamming_dist', val_hamming, epoch)
        log_metric('learning_rate', lr, epoch)
        log_metric('threshold', threshold, epoch)

        # ---------------------------------------------------------------------
        # SAVE SNAPSHOTS OF THE MODEL'S WEIGHTS (EVERY OTHER EPOCH)
        # ---------------------------------------------------------------------

        if epoch % 2 == 0:

            # Check if the appropriate directory for this run exists
            snapshot_dir = os.path.join('./weights/', log_name_formatted)
            if not os.path.exists(snapshot_dir):
                os.makedirs(snapshot_dir)

            # Save the weights for the current epoch ("snapshot")
            __ = [noise_source, distances, sample_size, epoch]
            weights_file_name = 'weights_{}_{}_{}_epoch-{:03d}.net'.format(*__)
            weights_file_path = os.path.join(snapshot_dir, weights_file_name)
            torch.save(model.state_dict(), weights_file_path)

        # ---------------------------------------------------------------------
        # REDUCE THE LEARNING RATE AND FUZZY ZONE THRESHOLD IF APPROPRIATE
        # ---------------------------------------------------------------------

        # Reduce Learning Rate if on Plateau
        scheduler.step(val_loss)

        # Get the minimum validation loss and Hamming distance
        min_val_loss = np.min(metrics['val_loss'])
        min_val_hamming = np.min(metrics['val_hamming'])

    #
    # -------------------------------------------------------------------------

    print('Finished Training!')
    writer.close()

    # -------------------------------------------------------------------------
    # AFTER TRAINING IS FINISHED, SAVE THE TRAINED MODEL
    # -------------------------------------------------------------------------

    print('Saving model...', end=' ')
    weights_file = ('./weights/timeseries_weights_{}_{}_{}_{:.1f}_FWHM.net'.
                    format(noise_source, distances, sample_size, threshold))
    torch.save(model.state_dict(), weights_file)
    print('Done!')

    # -------------------------------------------------------------------------
    # FINALLY, MAKE PREDICTIONS ON THE TEST SET AND SAVE THEM
    # -------------------------------------------------------------------------

    print('Start making predictions on the test sample...', end=' ')

    # Convert test data to numpy arrays that can be stored in an HDF file
    x_test = data_test.data_tensor.cpu().numpy()
    y_test = data_test.target_tensor.cpu().numpy()

    # Set up the data load for the test data
    data_loader_test = DataLoader(data_test, batch_size=batch_size)

    # Apply the net to the validation data and get the outputs
    test_outputs = apply_model(model, data_loader_test, as_numpy=False)

    # Get the true labels for the test data and calculate weights
    test_raw_labels = Variable(data_test.target_tensor, volatile=True)
    test_labels = get_labels(test_raw_labels, threshold)
    test_weights = get_weights(test_raw_labels, threshold)

    # Calculate the test loss
    test_loss = loss_function(y_pred=test_outputs,
                              y_true=test_labels,
                              weights=test_weights)
    test_loss = float(test_loss.data.cpu().numpy())

    # Calculate the test Hamming distance
    test_hamming = hamming_dist(y_pred=(test_outputs * test_weights),
                                y_true=(test_labels * test_weights))

    print('Done!')
    print('Saving predictions to HDF file...', end=' ')

    # Save the predictions of the model on the test set
    with h5py.File(pred_file_path, 'w') as file:

        # Store the data (input, label, network output) for test set
        file['x'] = x_test
        file['y_pred'] = test_outputs.data.cpu().numpy()
        file['y_true'] = y_test

        # Store the (averaged) loss and Hamming distance on the test set
        file['avg_loss'] = test_loss
        file['avg_hamming_distance'] = test_hamming

    print('Done!')
    print('TRAINING FINISHED! :)')
