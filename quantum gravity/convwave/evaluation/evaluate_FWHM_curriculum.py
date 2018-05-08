# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import os
import sys
import h5py
import torch
import torch.nn as nn

from collections import OrderedDict
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable

sys.path.insert(0, '../train/')
from models import TimeSeriesFCN

from IPython import embed


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def load_data_as_tensor_datasets(file_path, random_seed=42):

    # Set the seed for the random number generator
    np.random.seed(random_seed)

    # Read in the time series from the HDF file
    with h5py.File(file_path, 'r') as file:

        x = np.array(file['timeseries'])
        y = np.array(file['labels'])

    # Swap axes around to get to NCHW format
    x = np.swapaxes(x, 1, 3)
    x = np.swapaxes(x, 2, 3)
    x = np.squeeze(x)

    # Convert to torch Tensors
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()

    # Create TensorDatasets for training, test and validation
    tensor_dataset = TensorDataset(x, y)

    return tensor_dataset


# -----------------------------------------------------------------------------


def apply_model(model, data_loader, as_numpy=False):

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
        y_pred.append(outputs.cpu())

    # Concatenate the list of Variables to one Variable (this is faster than
    # concatenating all intermediate results) and make sure results are float
    y_pred = torch.cat(y_pred, dim=0).float().cuda()

    # If necessary, convert model outputs to numpy array
    if as_numpy:
        y_pred = y_pred.data.cpu().numpy()

    return y_pred


# -----------------------------------------------------------------------------


def get_labels(raw_labels, threshold):

    labels = torch.gt(raw_labels, threshold)
    return labels.float()


# -----------------------------------------------------------------------------


def loss_func(y_pred, y_true):

    # Set up the Binary Cross-Entropy term of the loss
    bce_loss = nn.BCELoss()

    # Check if CUDA is available to speed up computations
    if torch.cuda.is_available():
        bce_loss = bce_loss.cuda()

    # Calculate the loss
    loss = bce_loss(y_pred, y_true)

    # Return the result as a simple float number
    return float(loss.data.cpu().numpy())


# -----------------------------------------------------------------------------


def accuracy_func(y_true, y_pred):

    # Make sure y_pred is rounded to 0/1
    y_pred = torch.round(y_pred)

    result = torch.mean(torch.abs(y_true - y_pred), dim=1)
    result = torch.mean(result, dim=0)

    return 1 - float(result.data.cpu().numpy())


# -----------------------------------------------------------------------------
# MAIN PROGRAM
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    threshold = 0.0
    sample_size = '4k'

    # -------------------------------------------------------------------------
    # LOOP OVER THE DIFFERENT EVENT / DISTANCE RANGE COMBINATIONS
    # -------------------------------------------------------------------------

    for event in ['GW150914', 'GW151226', 'GW170104']:
        for dist in ['0100_0300', '0250_0500', '0400_0800', '0700_1200',
                     '1000_1700']:

            # -----------------------------------------------------------------
            # LOAD THE MODEL AND THE CORRECT WEIGHTS FILE
            # -----------------------------------------------------------------

            # Initialize the model
            model = TimeSeriesFCN()

            # Define the weights file we want to use for evaluation
            _ = ['..', 'train', 'weights', 'fwhm_curriculum',
                 'timeseries_weights_{}_{}_{}_{:.1f}_FWHM.net'.
                 format(event, dist, sample_size, threshold)]
            weights_file = os.path.join(*_)

            # Check if CUDA is available. If not, loading the weights is a bit
            # more cumbersome and we have to use some tricks
            if torch.cuda.is_available():
                model.float().cuda()
                model = torch.nn.DataParallel(model)
                model.load_state_dict(torch.load(weights_file))
            else:
                state_dict = torch.load(weights_file,
                                        map_location=lambda strge, loc: strge)
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)

            # -----------------------------------------------------------------
            # ACTUALLY PERFORM THE EVALUATION
            # -----------------------------------------------------------------

            print('NOW EVALUATING FWHM BASELINE FOR: {}, {}'.format(event,
                                                                    dist))

            # Load data into data tensor and data loader
            file_path = os.path.join('..', 'data', 'testing', 'timeseries',
                                     'testing_{}_{}_{}_FWHM.h5'.
                                     format(event, dist, sample_size))
            datatensor = load_data_as_tensor_datasets(file_path)
            dataloader = DataLoader(datatensor, batch_size=16)

            # Get the true labels we need for the comparison
            raw_labels = Variable(datatensor.target_tensor, volatile=True)
            if torch.cuda.is_available():
                raw_labels = raw_labels.cuda()
            labels = get_labels(raw_labels, threshold)

            # Get the predictions by applying the pre-trained net
            predictions = apply_model(model, dataloader)

            # Calculate the loss (averaged over the entire data set)
            loss = loss_func(y_pred=predictions,
                             y_true=labels)

            # Calculate the accuracy (averaged over the entire data set)
            accuracy = accuracy_func(y_pred=predictions,
                                     y_true=labels)

            # Print the results
            print('Loss: ... {:.3f}'.format(loss))
            print('Accuracy: {:.1f}%'.format(100 * accuracy))
            print()

        print(53 * '-')
        print()
