# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import torch
import h5py
import time
import argparse
import sys

from torch.utils.data import TensorDataset
from torch.autograd import Variable


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def create_weights(label, start_size=1000, end_size=200):
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


# -----------------------------------------------------------------------------


def hamming_dist(y_true, y_pred):
    """
    Calculate the Hamming distance between a given predicted label and the
    true label. Assumes inputs are torch Variables!

    Args:
        y_true (autograd.Variable): The true label
        y_pred (autograd.Variable): The predicted label

    Returns:
        (float): The Hamming distance between the two vectors
    """

    # Make sure y_pred is rounded to 0/1
    y_pred = torch.round(y_pred)

    result = torch.mean(torch.abs(y_true - y_pred), dim=1)
    result = torch.mean(result, dim=0)

    return float(result.data.cpu().numpy())


# -----------------------------------------------------------------------------


def progress_bar(current_value, max_value, start_time, end='', **kwargs):
    """
    Print the progress bar during training that contains all relevant
    information, i.e. number of epochs, percentage of processed mini-batches,
    elapsed time, estimated time remaining, as well as all metrics provided.

    Args:
        current_value: Current number of processed mini-batches
        max_value: Number of total mini-batches
        start_time: Absolute timestamp of the moment the epoch began
        end: How to end the line (i.e., start a new line or not)
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
        if type(value) is str:
            metrics.append("{}: {}".format(metric, value))
        else:
            if metric != 'lr':
                metrics.append("{}: {:.3f}".format(metric, value))
            else:
                metrics.append("{}: {:.8f}".format(metric, value))
    out += ' - '.join(metrics) + ' '

    # Actually write the finished progress bar to the command line
    sys.stdout.write(out + end)
    sys.stdout.flush()


def load_data_as_tensor_datasets(file_path, split_ratios=(0.8, 0.1, 0.1),
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

        x = np.array(file['timeseries'])
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


def apply_model(model, data_loader, as_numpy=False):
    """
    Take a model and a data loader, apply the model to the mini-batches from
    that dataloader, and return the results as a single Tensor / array.

    Args:
        model: A PyTorch model, i.e. in our case a FCN
        data_loader: A data loader which gives us the mini-batches
        as_numpy: If True, results are converted to a numpy array

    Returns:
        The outputs when the model is applied to the data of the data loader
    """

    # Initialize an empty array for our predictions
    y_pred = []

    # Loop over the test set (in mini-batches) to get the predictions
    for mb_idx, mb_data in enumerate(data_loader):

        # Get the inputs and wrap them in a PyTorch variable
        inputs, _ = mb_data
        inputs = Variable(inputs, volatile=True)

        # If CUDA is available, run everything on the GPU
        if torch.cuda.is_available():
            inputs = inputs.cuda()

        # Make predictions for the given mini-batch
        outputs = model.forward(inputs)
        outputs = outputs.view((outputs.size()[0], outputs.size()[-1]))

        # Stack that onto the previous predictions
        y_pred.append(outputs)

    # Concatenate the list of Variables to one Variable (this is faster than
    # concatenating all intermediate results) and make sure results are float
    y_pred = torch.cat(y_pred, dim=0).float()

    # If necessary, convert model outputs to numpy array
    if as_numpy:
        y_pred = y_pred.data.cpu().numpy()

    return y_pred


def get_current_lr(optimizer):
    """
    Retrieve the current learning rate from an optimizer.

    Args:
        optimizer (torch.optim.Optimizer): An instance of an optimizer class

    Returns:
        (float): The current learning rate of that optimizer
    """

    lr = None
    for param_group in optimizer.param_groups:
        lr = param_group['lr']

    return lr


def get_weights(labels, threshold):
    """
    Take a mini-batch of label vectors (i.e., signal envelopes) and a
    threshold, and return a vector that is 0 if 0 < label <= threshold, and
    1 otherwise, thus allowing us to make the signal parts below the
    threshold fuzzy for the computation of our metrics.

    Args:
        labels: A mini-batch of label vectors (signal envelopes / amplitudes)
        threshold: A number to use as a threshold on the signal amplitude

    Returns:
        A weights tensor (0 for fuzzy, 1 for non-fuzzy) that can be
        multiplied with the real and predicted labels before evaluating the
        loss and Hamming distance
    """

    weights = torch.eq(torch.gt(labels, threshold) *
                       torch.lt(labels, 1.2 * threshold), 0)

    return weights.float()


def get_labels(raw_labels, threshold):

    labels = torch.gt(raw_labels, threshold)
    return labels.float()


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class TrainingArgumentParser:

    def __init__(self):

        # Set up the parser
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
        self.parser = argparse.ArgumentParser(formatter_class=formatter_class)

        # Add command line options
        self.parser.add_argument('--batch-size',
                                 help='Batch size used for training',
                                 type=int,
                                 default=16)
        self.parser.add_argument('--description',
                                 help='Description of this run (optional)',
                                 type=str,
                                 default='No description available.')
        self.parser.add_argument('--distances',
                                 help='Distance range, e.g. "0100_0300"',
                                 type=str,
                                 default='0100_0300')
        self.parser.add_argument('--initial-lr',
                                 help='Initial learning rate',
                                 type=float,
                                 default=0.0001)
        self.parser.add_argument('--n-epochs',
                                 help='Number of epochs to train for',
                                 type=int,
                                 default=10)
        self.parser.add_argument('--noise-source',
                                 help='Where did the noise come from?',
                                 choices=['GW150914', 'GW151226',
                                          'GW170104', 'GAUSSIAN'],
                                 default='GW170104')
        self.parser.add_argument('--regularization-parameter',
                                 help='Weight parameter for TV regularization',
                                 type=float,
                                 default=0.00002)
        self.parser.add_argument('--sample-size',
                                 help='Sample length in seconds',
                                 type=str,
                                 default='4k')
        self.parser.add_argument('--threshold',
                                 help='Which threshold to apply for label '
                                      'creation (i.e., envelope > threshold)',
                                 type=float,
                                 default=0.0)
        self.parser.add_argument('--weights-file-name',
                                 help='Weight file to load for warm start',
                                 type=str,
                                 default=None)

    # -------------------------------------------------------------------------

    def parse_args(self):

        # Parse arguments and return them as a dict instead of Namespace
        return self.parser.parse_args().__dict__

    # -------------------------------------------------------------------------
