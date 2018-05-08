"""
Provides a number of functions for the sample generation process.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import sys
import json

import numpy as np
import pandas as pd
import h5py

from matplotlib import mlab
from scipy.signal import butter, filtfilt, medfilt, hilbert
from scipy.interpolate import interp1d


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def progress_bar(current_value, max_value, elapsed_time=0, bar_length=50):
    """
    Print a progress bar to the terminal to see how things are moving along.

    Args:
        current_value: The current number of spectrograms processed.
        max_value: The maximum number of spectrograms to be processed.
        elapsed_time: The time that has elapsed since the start of the script.
        bar_length: Maximum length of the bar.
    """

    # Construct the actual progress bar
    percent = float(current_value) / max_value
    full_bar = '=' * int(round(percent * bar_length))
    empty_bar = '-' * (bar_length - len(full_bar))

    # Calculate the estimated time remaining
    eta = elapsed_time / percent - elapsed_time

    # Collect the outputs and write them to stdout; move the carriage back
    # to the start of the line so that the progress bar is always updated.
    out = ("\r[{0}] {1}% ({2}/{3}) | {4:.1f}s elapsed | ETA: {5:.1f}s".
           format(full_bar + empty_bar, int(round(percent * 100)),
                  current_value, max_value, elapsed_time, eta))
    sys.stdout.write(out)
    sys.stdout.flush()


# -----------------------------------------------------------------------------


def apply_psd(signal_t, psd, sampling_rate=4096, apply_butter=True):
    """
    Take a signal in the time domain, and a precalculated Power Spectral
    Density, and color the signal according to the given PSD.

    Args:
        signal_t: A signal in time domain (i.e. a 1D numpy array)
        psd: A Power Spectral Density, e.g. calculated from the detector noise.
            Should be a function: psd(frequency)
        sampling_rate: Sampling rate of signal_t
        apply_butter: Whether or not to apply a Butterworth filter to the data.

    Returns: color_signal_t, the colored signal in the time domain.
    """

    # First set some parameters for computing power spectra
    signal_size = len(signal_t)
    delta_t = 1 / sampling_rate

    # Go into Fourier (frequency) space: signal_t -> signal_f
    frequencies = np.fft.rfftfreq(signal_size, delta_t)
    signal_f = np.fft.rfft(signal_t)

    # Divide by the given Power Spectral Density (PSD)
    # This is the 'whitening' = actually adding color
    color_signal_f = signal_f / (np.sqrt(psd(frequencies) / delta_t / 2))

    # Go back into time space: color_signal_f -> color_signal_t
    color_signal_t = np.fft.irfft(color_signal_f, n=signal_size)

    # In case we want to use a Butterworth-filter, here's how to do it:
    if apply_butter:

        # Define cut-off frequencies for the filter
        f_low = 42
        f_high = 800

        # Calculate Butterworth-filter and normalization
        numerator, denominator = butter(4, [f_low*2/4096, f_high*2/4096],
                                        btype="bandpass")
        normalization = np.sqrt((f_high - f_low) / (sampling_rate / 2))

        # Apply filter and normalize
        color_signal_t = filtfilt(numerator, denominator, color_signal_t)
        color_signal_t = color_signal_t / normalization

    return color_signal_t


# -----------------------------------------------------------------------------


def get_psd(real_strain, sampling_rate=4096):
    """
    Take a detector recording and calculate the Power Spectral Density (PSD).

    Args:
        real_strain: The detector recording to be used.
        sampling_rate: The sampling rate (in Hz) of the recording

    Returns:
        psd: The Power Spectral Density of the detector recordings
    """

    # Define some constants
    nfft = 2 * sampling_rate  # Bigger values yield better resolution?

    # Use matplotlib.mlab to calculate the PSD from the real strain
    power_spectrum, frequencies = mlab.psd(real_strain,
                                           NFFT=nfft,
                                           Fs=sampling_rate)

    # Interpolate it linearly, so we can re-sample the spectrum arbitrarily
    psd = interp1d(frequencies, power_spectrum)

    return psd


# -----------------------------------------------------------------------------


def chirp_mass(mass1, mass2):
    """
    Takes two masses and calculates the corresponding chirpmass.
    Args:
        mass1: Mass 1
        mass2: Mass 2

    Returns:
        chirpmass: The chirpmass that corresponds to mass1, mass2
    """

    return (mass1 * mass2) ** (3 / 5) / (mass1 + mass2) ** (1 / 5)


# -----------------------------------------------------------------------------


def get_waveforms_as_dataframe(waveforms_path):
    """
    Take an HDF file containing pre-generated waveforms (as by the
    waveform_generator.py in this repository) and extract the relevant
    information (waveform, mass 1, mass 2, chirpmass, distance) into a
    pandas DataFrame for convenient access.

    Args:
        waveforms_path: The path to the HDF file containing the waveforms

    Returns:
        dataframe: A pandas DataFrame containing all valid waveforms and their
            corresponding masses, chirpmasses and distances.
    """

    # Read in the actual waveforms, the config string (and parse from JSON),
    # and the indices of the failed waveforms
    with h5py.File(waveforms_path, 'r') as file:
        waveforms = np.array(file['waveforms'])
        config = json.loads(file['config'].value.astype('str'))['injections']
        failed_idx = np.array(file['failed'])

    # Create a Pandas DataFrame containing only the relevant columns from the
    # config string (other columns are all trivial at this point)
    columns = ['distance', 'mass1', 'mass2']
    dataframe = pd.DataFrame(config, columns=columns)

    # Add columns for the actual waveforms and the chirp masses
    dataframe['waveform'] = list(waveforms)
    dataframe['chirpmass'] = dataframe.apply(lambda row: chirp_mass(row.mass1,
                                                                    row.mass2),
                                             axis=1)

    # Drop the rows with the failed waveforms, and reset the index
    # noinspection PyUnresolvedReferences
    dataframe = dataframe.drop(list(failed_idx)).reset_index(drop=True)

    # Resort columns to order them alphabetically
    dataframe = dataframe[sorted(dataframe.columns)]

    # Return the final DataFrame
    return dataframe


# -----------------------------------------------------------------------------


def get_start_end_idx(waveform):
    """
    Take a raw waveform and return the indices when the signal actually
    begins and ends, i.e. the indices of the first non-zero elements in the
    (reversed) waveform.

    Args:
        waveform: A raw waveform, i.e. one that still is zero-padded.

    Returns:
        start, end: The indices where the signal begins / ends.

    """

    # Initialize empty variables for the beginning / end
    start = None
    end = None

    # Find the start of the signal
    for j in range(len(waveform)):
        if waveform[j] != 0:
            start = j
            break

    # Find the end of the signal
    for j in sorted(range(len(waveform)), reverse=True):
        if waveform[j] != 0:
            end = j
            break

    return start, end


# -----------------------------------------------------------------------------


def get_envelope(signal):
    # Pad the signal with zeros at the beginning and end to reduce edge effects
    padded_signal = np.pad(signal, 100, 'constant', constant_values=0)

    # Calculate the raw envelope using the Hilbert transformation
    analytic_signal = hilbert(padded_signal)
    amplitude_envelope = np.abs(analytic_signal)

    # Smoothen the envelope using a median filter and a rolling average
    smooth = amplitude_envelope
    smooth[0:200] = medfilt(smooth[0:200], kernel_size=25)
    smooth = np.convolve(smooth, np.ones(10), mode='same') / 10

    # Remove the zero padding again to match the original signal length
    result = smooth[100:-100]

    return result


# -----------------------------------------------------------------------------


def resample_vector(vector, new_length):

    interpolation = interp1d(range(len(vector)), vector, 'linear')
    grid = np.linspace(0, len(vector)-1, new_length)

    return np.round(interpolation(grid))


# -----------------------------------------------------------------------------


def snr_from_results_list(results_list, ifo, max_n_injections):
    results = []
    for entry in results_list:
        if not entry:
            results.append(max_n_injections * [np.nan])
        else:
            foo = [_[ifo] for _ in entry]
            while len(foo) < max_n_injections:
                foo.append(np.nan)
            results.append(foo)
    return np.array(results)
