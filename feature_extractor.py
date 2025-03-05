"""
This module extract salsa-lite and salsa-ipd feature for microphone array format.
"""
import os
import shutil

import h5py
import librosa
import numpy as np
import yaml
from timeit import default_timer as timer
from tqdm import tqdm


def salsa(data, data_config: str = 'configs/tnsse2021_salsa_lite_feature_config.yml'):
    """
    Extract salsa_lite and salsa_ipd features:
        salsa_lite: log-linear spectrogram + normalized interchannel phase difference
        salsa_ipd:  log-linear spectrogram + interchannel phase difference
    The frequency range of log-linear spectrogram is 0 to 9kHz.
    :param data_config: Path to data config file.
    :param feature_type: Choices are 'salsa_lite', 'salsa_ipd'
    :param task: 'feature_scaler': extract feature and scaler, 'feature': only extract feature, 'scaler': only extract
        scaler.
    """
    # Load data config files
    with open(data_config, 'r') as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Parse config file
    audio_format = cfg['data']['format']
    fs = cfg['data']['fs']
    n_fft = cfg['data']['n_fft']
    hop_length = cfg['data']['hop_len']
    win_length = cfg['data']['win_len']

    # Doa info
    n_mics = 4
    fmin_doa = cfg['data']['fmin_doa']
    fmax_doa = cfg['data']['fmax_doa']
    fmax_doa = np.min((fmax_doa, fs // 2))
    n_bins = n_fft // 2 + 1
    lower_bin = np.int(np.floor(fmin_doa * n_fft / np.float(fs)))  # 512: 1; 256: 0
    upper_bin = np.int(np.floor(fmax_doa * n_fft / np.float(fs)))  # 9000Hz: 512: 192, 256: 96
    lower_bin = np.max((1, lower_bin))

    # Cutoff frequency for spectrograms
    fmax = 9000  # Hz
    cutoff_bin = np.int(np.floor(fmax * n_fft / np.float(fs)))  # 9000 Hz, 512 nfft: cutoff_bin = 192
    assert upper_bin <= cutoff_bin, 'Upper bin for spatial feature is higher than cutoff bin for spectrogram!'

    # Normalization factor for salsa_lite
    c = 343
    delta = 2 * np.pi * fs / (n_fft * c)
    freq_vector = np.arange(n_bins)
    freq_vector[0] = 1
    freq_vector = freq_vector[:, None, None]  # n_bins x 1 x 1

    # Get feature descriptions
    feature_description = '{}fs_{}nfft_{}nhop_{}fmaxdoa'.format(fs, n_fft, hop_length, int(fmax_doa))
    print('Feature description: {}'.format(feature_description))
    # Extract features
    full_audio_fn = data
    audio_input, _ = librosa.load(full_audio_fn, sr=fs, mono=False, dtype=np.float32)
    # Compute stft
    log_specs = []
    for imic in np.arange(n_mics):
        stft = librosa.stft(y=np.asfortranarray(audio_input[imic, :]), n_fft=n_fft, hop_length=hop_length,
                            center=True, window='hann', pad_mode='reflect')
        if imic == 0:
            n_frames = stft.shape[1]
            X = np.zeros((n_bins, n_frames, n_mics), dtype='complex')  # (n_bins, n_frames, n_mics)
        X[:, :, imic] = stft
        # Compute log linear power spectrum
        spec = (np.abs(stft) ** 2).T
        log_spec = librosa.power_to_db(spec, ref=1.0, amin=1e-10, top_db=None)
        log_spec = np.expand_dims(log_spec, axis=0)
        log_specs.append(log_spec)
    log_specs = np.concatenate(log_specs, axis=0)  # (n_mics, n_frames, n_bins)

    # Compute spatial feature
    phase_vector = np.angle(X[:, :, 1:] * np.conj(X[:, :, 0, None]))

    phase_vector = phase_vector / (delta * freq_vector)
    phase_vector = np.transpose(phase_vector, (2, 1, 0))  # (n_mics, n_frames, n_bins)
    # Crop frequency
    log_specs = log_specs[:, :, lower_bin:cutoff_bin]
    phase_vector = phase_vector[:, :, lower_bin:cutoff_bin]
    phase_vector[:, :, upper_bin:] = 0

    # Stack features
    audio_feature = np.concatenate((log_specs, phase_vector), axis=0)
    return audio_feature
