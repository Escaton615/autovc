import os
import pickle
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import get_window
from librosa.filters import mel
from numpy.random import RandomState


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def pySTFT(x, fft_length=1024, hop_length=256):
    # x = np.pad(x, int(fft_length//2), mode='reflect')

    noverlap = fft_length - hop_length
    shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // hop_length, fft_length)
    strides = x.strides[:-1] + (hop_length * x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)

    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    print(result.dtype)
    return np.abs(result)


mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))
b, a = butter_highpass(30, 16000, order=5)


def wav2mel(path, mel_length=0):
    # Read audio file
    x, fs = sf.read(path)
    # Remove drifting noise
    y = signal.filtfilt(b, a, x)
    # Ddd a little random noise for model roubstness
    wav = y * 0.96 + (np.random.rand(y.shape[0]) - 0.5) * 1e-06
    if mel_length > 0:
        clip_length = mel_length * 256 + 768
        clip_start_idx = np.random.randint(0, len(wav) - 1 - clip_length)
        wav = wav[clip_start_idx: clip_start_idx + clip_length]
        y = y[clip_start_idx: clip_start_idx + clip_length]

    # Compute spect
    D = pySTFT(wav).T
    # Convert to mel and normalize
    D_mel = np.dot(D, mel_basis)
    D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
    S = np.clip((D_db + 100) / 100, 0, 1)
    if mel_length > 0:
        return S, y
    else:
        return S


if __name__ == "__main__":

    # audio file directory
    rootDir = '/media/dt4/DG_1TB_1/manwoman_vocals_16k_rearranged/'
    # spectrogram directory
    targetDir = '/home/dt4/mel_mw163/'

    dirName, subdirList, _ = next(os.walk(rootDir))
    print('Found directory: %s' % dirName)

    for subdir in sorted(subdirList):
        print(subdir)
        if not os.path.exists(os.path.join(targetDir, subdir)):
            os.makedirs(os.path.join(targetDir, subdir))
        _, _, fileList = next(os.walk(os.path.join(dirName, subdir)))
        prng = RandomState(int(subdir[1:]))
        for fileName in sorted(fileList):
            # Read audio file
            # x, fs = sf.read(os.path.join(dirName,subdir,fileName))
            # # Remove drifting noise
            # y = signal.filtfilt(b, a, x)
            # # Ddd a little random noise for model roubstness
            # wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06
            # # Compute spect
            # D = pySTFT(wav).T
            # # Convert to mel and normalize
            # D_mel = np.dot(D, mel_basis)
            # D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
            S = wav2mel(os.path.join(dirName, subdir, fileName))  # np.clip((D_db + 100) / 100, 0, 1)
            # save spect
            np.save(os.path.join(targetDir, subdir, fileName[:-4]),
                    S.astype(np.float32), allow_pickle=False)
