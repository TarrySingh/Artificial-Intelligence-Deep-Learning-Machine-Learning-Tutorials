# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import abc
import numpy as np
import argparse

from librosa.feature import melspectrogram
from librosa import logamplitude
import warnings

from sample_generation_tools import apply_psd, get_start_end_idx, \
    get_envelope, resample_vector


# -----------------------------------------------------------------------------
# CLASS FOR PARSING COMMAND LINE ARGUMENTS FOR THE GENERATORS
# -----------------------------------------------------------------------------

class CustomArgumentParser:

    def __init__(self):

        # Set up the parser
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
        self.parser = argparse.ArgumentParser(formatter_class=formatter_class)

        # Add command line options
        self.parser.add_argument('--n-samples',
                                 help='Number of samples to generate',
                                 type=int,
                                 default=64)
        self.parser.add_argument('--sample-length',
                                 help='Sample length in seconds',
                                 type=int,
                                 default=12)
        self.parser.add_argument('--sampling-rate',
                                 help='Sampling rate in Hz',
                                 type=int,
                                 default=4096)
        self.parser.add_argument('--max-n-injections',
                                 help='Max number of injections per sample',
                                 type=int,
                                 default=2)
        self.parser.add_argument('--loudness',
                                 help='Scaling factor for injections',
                                 type=float,
                                 default=1.0)
        self.parser.add_argument('--pad',
                                 help='Noise padding to avoid spectral '
                                      'leakage (lengths in seconds)',
                                 type=float,
                                 default=3.0)
        self.parser.add_argument('--data-path',
                                 help='Path of the data directory',
                                 type=str,
                                 default='../data/')
        self.parser.add_argument('--waveforms-file',
                                 help='Name of the file containing the '
                                      'pre-computed waveforms',
                                 type=str,
                                 default='samples_dist_100_300.h5')
        self.parser.add_argument('--output-file',
                                 help='Name of the ouput HDF file',
                                 type=str,
                                 default='training_samples.h5')
        self.parser.add_argument('--noise-type',
                                 help='Type of noise used for injections',
                                 choices=['gaussian', 'real'],
                                 default='real')
        self.parser.add_argument('--strain-file',
                                 help='Strain from which event to use',
                                 choices=['GW150914', 'GW151226', 'GW170104'],
                                 default='GW150914')
        self.parser.add_argument('--sample-type',
                                 help='Type of sample to create',
                                 choices=['timeseries', 'spectrograms'],
                                 default='timeseries')
        self.parser.add_argument('--use-type',
                                 help='Use data for training or testing?',
                                 choices=['training', 'testing'],
                                 default='training')

    # -------------------------------------------------------------------------

    def parse_args(self):

        # Parse arguments and return them as a dict instead of Namespace
        return self.parser.parse_args().__dict__

    # -------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# BASE CLASS FOR THE SPECTROGRAM AND THE TIME SERIES SAMPLE GENERATOR
# -----------------------------------------------------------------------------

class SampleGenerator:

    def __init__(self, sample_length, sampling_rate, max_n_injections,
                 waveforms, real_strains, white_strain_std, psds,
                 noise_type, max_delta_t=0.01, loudness=1.0, pad=3.0,
                 event_position=None):

        # Store all parameters passed as arguments
        self.sample_length = sample_length
        self.sampling_rate = sampling_rate
        self.max_n_injections = max_n_injections
        self.waveforms = waveforms
        self.real_strains = real_strains
        self.white_strain_std = white_strain_std
        self.pad = pad
        self.psds = psds
        self.noise_type = noise_type
        self.max_delta_t = max_delta_t
        self.loudness = loudness
        self.event_position = event_position

        # Initialize all other class attributes
        self.chirpmasses = None
        self.distances = None
        self.labels = None
        self.noises = None
        self.positions = None
        self.signals = None
        self.snrs = None
        self.strains = None

        # Randomly choose the number of injections for this sample
        self.n_injections = np.random.randint(0, self.max_n_injections + 1)

        # Add padding so that we can remove the fringe effects due to applying
        # the PSD and calculating the spectrogram later on
        self.length = (self.sample_length + 2 * self.pad) * self.sampling_rate
        self.length = int(self.length)

        # Create a random time difference between the signals for H1 and L1
        self.delta_t = np.random.uniform(-1 * max_delta_t, max_delta_t)
        self.offset = int(self.delta_t * self.sampling_rate)

        # Create the noises (self.noise_type determines with we use Gaussian
        # noise or real noise from the detector recordings)
        self.noises = self._make_noises()

        # Create the signals and the label vectors by making some injections
        # NOTE: In case of the spectrograms, the labels need to be re-sized to
        # match the length of the spectrogram!
        self.signals, self.labels, self.chirpmasses, self.distances, \
            self.snrs = self._make_signals()

        # Calculate the strains as the sum of the noises and the signals
        self.strains = self._make_strains()

    # -------------------------------------------------------------------------

    def _make_noises(self):

        noises = dict()

        # In case we are working with simulated noise, just create some
        # Gaussian noise with the correct length
        if self.noise_type == 'gaussian':

            noises['H1'] = np.random.normal(0, 1, self.length)
            noises['L1'] = np.random.normal(0, 1, self.length)

        # If we are using real noise, select some random subset of the
        # provided noise, i.e. a random piece of the real strain data
        elif self.noise_type == 'real':

            # Find maximum starting position
            max_pos = dict()
            max_pos['H1'] = len(self.real_strains['H1']) - self.length
            max_pos['L1'] = len(self.real_strains['L1']) - self.length

            # Randomly find the starting positions
            start = dict()
            start['H1'] = int(np.random.uniform(0, max_pos['H1']))
            start['L1'] = int(np.random.uniform(0, max_pos['L1']))

            # Make sure no real event is contained in the noise we select
            while ((self.event_position - self.sample_length) <
                   (start['H1'] / self.sampling_rate) <
                   (self.event_position + self.sample_length)):
                start['H1'] = int(np.random.uniform(0, max_pos['H1']))
            while ((self.event_position - self.sample_length) <
                   (start['L1'] / self.sampling_rate) <
                   (self.event_position + self.sample_length)):
                start['L1'] = int(np.random.uniform(0, max_pos['L1']))

            # Find the end positions
            end = dict()
            end['H1'] = int(start['H1'] + self.length)
            end['L1'] = int(start['L1'] + self.length)

            # Select random chunks of the real detector recording as the noise
            noises['H1'] = self.real_strains['H1'][start['H1']:end['H1']]
            noises['L1'] = self.real_strains['L1'][start['L1']:end['L1']]

        return noises

    # -------------------------------------------------------------------------

    def _make_signals(self):

        # Initialize empty vectors for the signal and the labels
        signals = {'H1': np.zeros(self.length), 'L1': np.zeros(self.length)}
        labels = {'H1': np.zeros(self.length), 'L1': np.zeros(self.length)}
        chirpmasses = {'H1': np.zeros(self.length),
                       'L1': np.zeros(self.length)}
        distances = {'H1': np.zeros(self.length), 'L1': np.zeros(self.length)}

        # Get the length of a single raw (!) waveform
        waveform_length = (len(self.waveforms.iloc[0]['waveform']) /
                           self.sampling_rate)

        # Calculate the start positions of the injections
        spacing = self.sample_length - self.n_injections * waveform_length
        spacing = spacing / (self.n_injections + 1)
        self.positions = [(self.pad + spacing + i * (waveform_length +
                           spacing)) * self.sampling_rate for i in
                          range(self.n_injections)]

        # Empty list to keep track of the SNRs we are calculating
        snrs = []

        # Loop over all injections to be made
        for inj_number in range(self.n_injections):

            # -----------------------------------------------------------------
            # Select random waveform and pre-process
            # -----------------------------------------------------------------

            # Randomly select a row from the waveforms DataFrame
            waveform_idx = np.random.randint(len(self.waveforms))
            waveform_row = self.waveforms.iloc[waveform_idx]

            # Extract the chirpmass and distance from that row
            chirpmass = waveform_row['chirpmass']
            distance = waveform_row['distance']

            # Get the raw waveform from that row and then remove the
            # zero-padding from it to get the pure waveform
            start, end = get_start_end_idx(waveform_row['waveform'])
            pure_waveform = waveform_row['waveform'][start:end]

            # If the pure waveform is at least 1 second long, we randomize its
            # length to something between 1 and <pure_waveform_length> seconds:
            pure_waveform_length = len(pure_waveform) / self.sampling_rate
            if pure_waveform_length > 1:
                cut_off = np.random.uniform(0, pure_waveform_length - 1)
                pure_waveform = pure_waveform[int(cut_off*self.sampling_rate):]

            # Now set the pure waveform as the waveform for H1 and L1. This
            # is necessary, because in the next step we (possibly) apply the
            # PSD, which is different for the two detectors.
            waveform = dict()
            waveform['H1'] = self.loudness * pure_waveform
            waveform['L1'] = self.loudness * pure_waveform

            # If we are using simulated Gaussian noise, we have to apply the
            # PSD directly to the waveform(s)
            if self.noise_type == 'gaussian':

                # Apply the Power Spectral Density to create a colored signal
                waveform['H1'] = apply_psd(waveform['H1'], self.psds['H1'])
                waveform['L1'] = apply_psd(waveform['L1'], self.psds['L1'])

                # Cut off spectral leakage that is due to the Fourier Transform
                waveform['H1'] = waveform['H1'][100:-50]
                waveform['L1'] = waveform['L1'][100:-50]

            # -----------------------------------------------------------------
            # Calculate envelopes and start / end positions for injections
            # -----------------------------------------------------------------

            # Now calculate the envelopes of these waveforms
            waveform_envelope = dict()
            waveform_envelope['H1'] = get_envelope(waveform['H1'])
            waveform_envelope['L1'] = get_envelope(waveform['L1'])

            # Calculate absolute starting positions of the injections
            start_pos = dict()
            start_pos['H1'] = int(self.positions[inj_number])
            start_pos['L1'] = int(self.positions[inj_number] + self.offset)

            # Calculate absolute waveform lengths of the injections
            waveform_length = dict()
            waveform_length['H1'] = len(waveform['H1'])
            waveform_length['L1'] = len(waveform['L1'])

            # Calculate the absolute end position of the injection
            end_pos = dict()
            end_pos['H1'] = int(start_pos['H1'] + waveform_length['H1'])
            end_pos['L1'] = int(start_pos['L1'] + waveform_length['L1'])

            # -----------------------------------------------------------------
            # Make injections and create label vectors
            # -----------------------------------------------------------------

            # Make the injection, i.e. add the waveform to the signal
            signals['H1'][start_pos['H1']:end_pos['H1']] += waveform['H1']
            signals['L1'][start_pos['L1']:end_pos['L1']] += waveform['L1']

            # Create the (normalized) envelope vector
            labels['H1'][start_pos['H1']:end_pos['H1']] += \
                (waveform_envelope['H1'] / np.max(waveform_envelope['H1']))
            labels['L1'][start_pos['L1']:end_pos['L1']] += \
                (waveform_envelope['L1'] / np.max(waveform_envelope['L1']))

            # Create the chirpmasses vector
            chirpmasses['H1'][start_pos['H1']:end_pos['H1']] += chirpmass
            chirpmasses['L1'][start_pos['L1']:end_pos['L1']] += chirpmass

            # Create the distances vector
            distances['H1'][start_pos['H1']:end_pos['H1']] += distance
            distances['L1'][start_pos['L1']:end_pos['L1']] += distance

            # -----------------------------------------------------------------
            # Calculate the SNRs for this injection
            # -----------------------------------------------------------------

            # Whiten the waveforms
            white_waveform = dict()
            white_waveform['H1'] = apply_psd(waveform['H1'], self.psds['H1'])
            white_waveform['L1'] = apply_psd(waveform['L1'], self.psds['L1'])

            # Get the amplitude maximum of the whitened signal
            max_signal = dict()
            max_signal['H1'] = np.max(np.abs(white_waveform['H1']))
            max_signal['L1'] = np.max(np.abs(white_waveform['L1']))

            # Calculate the peak signal-to-noise ratio
            snr = {'H1': max_signal['H1'] / self.white_strain_std['H1'],
                   'L1': max_signal['L1'] / self.white_strain_std['L1']}
            snrs.append(snr)

        return signals, labels, chirpmasses, distances, snrs

    # -------------------------------------------------------------------------

    def _make_strains(self):

        strains = dict()
        strains['H1'] = self.noises['H1'] + self.signals['H1']
        strains['L1'] = self.noises['L1'] + self.signals['L1']

        # If we are using real noise, we have to apply the PSD to the sum of
        # noise and signal to 'whiten' the strain:
        if self.noise_type == 'real':

            # Apply the Power Spectral Density to whiten
            strains['H1'] = apply_psd(strains['H1'], self.psds['H1'])
            strains['L1'] = apply_psd(strains['L1'], self.psds['L1'])

            self.noises['H1'] = apply_psd(self.noises['H1'], self.psds['H1'])
            self.noises['L1'] = apply_psd(self.noises['L1'], self.psds['L1'])
            self.signals['H1'] = apply_psd(self.signals['H1'],
                                           self.psds['H1'])
            self.signals['L1'] = apply_psd(self.signals['L1'],
                                           self.psds['L1'])

        return strains

    # -------------------------------------------------------------------------

    @abc.abstractmethod
    def _remove_padding(self):
        raise NotImplementedError()

    # -------------------------------------------------------------------------

    def get_label(self):

        return np.maximum(self.labels['H1'], self.labels['L1'])

    # -------------------------------------------------------------------------

    def get_chirpmass(self):

        return np.maximum(self.chirpmasses['H1'], self.chirpmasses['L1'])

    # -------------------------------------------------------------------------

    def get_distance(self):

        return np.maximum(self.distances['H1'], self.distances['L1'])

    # -------------------------------------------------------------------------

    def get_snr(self):

        return self.snrs

    # -------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# CLASS TO GENERATE SPECTROGRAM SAMPLES
# -----------------------------------------------------------------------------

class Spectrogram(SampleGenerator):

    def __init__(self, sample_length, sampling_rate, max_n_injections,
                 waveforms, real_strains, white_strain_std, psds, noise_type,
                 max_delta_t=0.01, loudness=1.0, pad=3.0, event_position=None):

        # Inherit from the SampleGenerator base class
        super().__init__(sample_length=sample_length,
                         sampling_rate=sampling_rate,
                         max_n_injections=max_n_injections,
                         waveforms=waveforms,
                         real_strains=real_strains,
                         white_strain_std=white_strain_std,
                         psds=psds,
                         noise_type=noise_type,
                         max_delta_t=max_delta_t,
                         loudness=loudness,
                         pad=pad,
                         event_position=event_position)

        # Add a variable for the spectrograms
        self.spectrograms = None

        # Calculate a spectrogram from the strain
        self.spectrograms = self._make_spectrograms()

        # Calculate the labels for this spectrogram
        label_lengths = {'H1': self.spectrograms['H1'].shape[1],
                         'L1': self.spectrograms['L1'].shape[1]}
        self.labels, self.chirpmasses, self.distances = \
            self._rescale_labels(new_lengths=label_lengths)

        # Finally, remove the padding again
        self.spectrograms, self.labels, self.chirpmasses, self.distances = \
            self._remove_padding()

    # -------------------------------------------------------------------------

    def _rescale_labels(self, new_lengths):

        # Rescale the labels to the length of the spectrogram using a
        # linear interpolation that is evaluated on a grid
        """
        labels = {'H1': resample_vector(self.labels['H1'], new_lengths['H1']),
                  'L1': resample_vector(self.labels['L1'], new_lengths['L1'])}
        """
        chirpmasses = {'H1': resample_vector(self.chirpmasses['H1'],
                                             new_lengths['H1']),
                       'L1': resample_vector(self.chirpmasses['L1'],
                                             new_lengths['L1'])}
        distances = {'H1': resample_vector(self.distances['H1'],
                                           new_lengths['H1']),
                     'L1': resample_vector(self.distances['L1'],
                                           new_lengths['L1'])}

        warnings.warn('Labels return by this method are not signal '
                      'envelopes, but just 0/1 for (no) injection present!')

        # FIXME: Linear interpolation seems to break on label scale (10^-21)
        # This is a dirty hack to get at least some kind of working labels
        labels = {'H1': np.fromiter(map(lambda x: x > 0, chirpmasses['H1']),
                                    dtype=int),
                  'L1': np.fromiter(map(lambda x: x > 0, chirpmasses['H1']),
                                    dtype=int)}

        return labels, chirpmasses, distances

    # -------------------------------------------------------------------------

    def _make_spectrograms(self):

        # Essentially curry the melspectrogram() function of librosa, because
        # we need to call it twice and this is just more readable
        def make_spectrogram(strain):
            return melspectrogram(strain, sr=4096, n_fft=1024, hop_length=64,
                                  n_mels=64, fmin=1, fmax=600)

        # Calculate the pure spectrograms
        spectrograms = dict()
        spectrograms['H1'] = make_spectrogram(self.strains['H1'])
        spectrograms['L1'] = make_spectrogram(self.strains['L1'])

        # Make the spectrograms log-amplitude
        spectrograms['H1'] = logamplitude(spectrograms['H1'])
        spectrograms['L1'] = logamplitude(spectrograms['L1'])

        return spectrograms

    # -------------------------------------------------------------------------

    def _remove_padding(self):

        # Get the lengths of the spectrograms
        lengths = dict()
        lengths['H1'] = self.spectrograms['H1'].shape[1]
        lengths['L1'] = self.spectrograms['L1'].shape[1]

        # Get the start of the "inner part" with the injections
        start = dict()
        start['H1'] = int((lengths['H1'] / self.length) * self.sampling_rate
                          * self.pad)
        start['L1'] = int((lengths['H1'] / self.length) * self.sampling_rate
                          * self.pad)

        # Get the end of the "inner part" with the injections
        end = dict()
        end['H1'] = -start['H1']
        end['L1'] = -start['L1']

        # For the spectrograms, only select the "inner part"
        spectrograms = dict()
        spectrograms['H1'] = self.spectrograms['H1'][:, start['H1']:end['H1']]
        spectrograms['L1'] = self.spectrograms['L1'][:, start['L1']:end['L1']]

        # For the label vectors, only select the "inner part"
        labels = dict()
        labels['H1'] = self.labels['H1'][start['H1']:end['H1']]
        labels['L1'] = self.labels['L1'][start['L1']:end['L1']]

        # For the chirpmass vectors, only select the "inner part"
        chirpmasses = dict()
        chirpmasses['H1'] = self.chirpmasses['H1'][start['H1']:end['H1']]
        chirpmasses['L1'] = self.chirpmasses['L1'][start['L1']:end['L1']]

        # For the distances vectors, only select the "inner part"
        distances = dict()
        distances['H1'] = self.distances['H1'][start['H1']:end['H1']]
        distances['L1'] = self.distances['L1'][start['L1']:end['L1']]

        return spectrograms, labels, chirpmasses, distances

    # -------------------------------------------------------------------------

    def get_spectrograms(self):

        # Stack the spectrograms. This produces the NHWC, or "channels last"
        # format, which is the standard for keras (but not for PyTorch!)
        return np.dstack((self.spectrograms['H1'], self.spectrograms['L1']))


# -----------------------------------------------------------------------------
# CLASS TO GENERATE TIME SERIES SAMPLES
# -----------------------------------------------------------------------------

class TimeSeries(SampleGenerator):

    def __init__(self, sample_length, sampling_rate, max_n_injections,
                 waveforms, real_strains, white_strain_std, psds, noise_type,
                 max_delta_t=0.01, loudness=1.0, pad=3.0, event_position=None):

        # Inherit from the SampleGenerator base class
        super().__init__(sample_length=sample_length,
                         sampling_rate=sampling_rate,
                         max_n_injections=max_n_injections,
                         waveforms=waveforms,
                         real_strains=real_strains,
                         white_strain_std=white_strain_std,
                         psds=psds,
                         noise_type=noise_type,
                         max_delta_t=max_delta_t,
                         loudness=loudness,
                         pad=pad,
                         event_position=event_position)

        # Remove the padding again
        self.strains, self.signals, self.noises, self.labels, \
            self.chirpmasses, self.distances = self._remove_padding()

        # Down-sample to 2048 Hz
        self.strains, self.signals, self.noises, self.labels, \
            self.chirpmasses, self.distances = self._downsample()

    # -------------------------------------------------------------------------

    def _remove_padding(self):

        pad_length = int(self.pad * self.sampling_rate)

        # Unpad the strains
        strains = dict()
        strains['H1'] = self.strains['H1'][pad_length:-pad_length]
        strains['L1'] = self.strains['L1'][pad_length:-pad_length]

        # Unpad the signals
        signals = dict()
        signals['H1'] = self.signals['H1'][pad_length:-pad_length]
        signals['L1'] = self.signals['L1'][pad_length:-pad_length]

        # Unpad the noises
        noises = dict()
        noises['H1'] = self.noises['H1'][pad_length:-pad_length]
        noises['L1'] = self.noises['L1'][pad_length:-pad_length]

        # Unpad the labels
        labels = dict()
        labels['H1'] = self.labels['H1'][pad_length:-pad_length]
        labels['L1'] = self.labels['L1'][pad_length:-pad_length]

        # Unpad the chirpmasses
        chirpmasses = dict()
        chirpmasses['H1'] = self.chirpmasses['H1'][pad_length:-pad_length]
        chirpmasses['L1'] = self.chirpmasses['L1'][pad_length:-pad_length]

        # Unpad the distances
        distances = dict()
        distances['H1'] = self.distances['H1'][pad_length:-pad_length]
        distances['L1'] = self.distances['L1'][pad_length:-pad_length]

        return strains, signals, noises, labels, chirpmasses, distances

    # -------------------------------------------------------------------------

    def _downsample(self):
        """
        Downsample the strains and the label, chirpmass and distance vectors
        to half their frequency, i.e. usually from 4096 Hz to 2048 Hz.

        Returns:
            strains: Downsampled version of the strains
            labels: Downsampled version of the label vectors
            chirpmasses: Downsampled version of the chirpmass vectors
            distances: Downsampled version of the distance vectors
        """

        # TODO: Make downsampling more generic, i.e. to arbitrary frequency?
        strains = {'H1': self.strains['H1'][::2],
                   'L1': self.strains['L1'][::2]}
        labels = {'H1': self.labels['H1'][::2],
                  'L1': self.labels['L1'][::2]}
        signals = {'H1': self.signals['H1'][::2],
                   'L1': self.signals['L1'][::2]}
        noises = {'H1': self.noises['H1'][::2],
                  'L1': self.noises['L1'][::2]}
        chirpmasses = {'H1': self.chirpmasses['H1'][::2],
                       'L1': self.chirpmasses['L1'][::2]}
        distances = {'H1': self.distances['H1'][::2],
                     'L1': self.distances['L1'][::2]}

        return strains, signals, noises, labels, chirpmasses, distances

    # -------------------------------------------------------------------------

    def get_timeseries(self):

        return np.dstack((self.strains['H1'], self.strains['L1']))
