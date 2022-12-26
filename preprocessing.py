from math import floor
import wave
import math
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def load(filename):
    ifile = wave.open(filename)
    samples = ifile.getnframes()
    raudio = ifile.readframes(samples)
    sr = ifile.getframerate()
    np_int16 = np.frombuffer(raudio, dtype=np.int16)
    signal = np_int16[::2].astype(np.float32)
    return signal, sr


def stft(signal, sr):
    offset = 0
    nfft = 2048
    hop_size = 512
    nsegs = math.ceil(len(signal) / hop_size)

    hanning = np.hanning(nfft)
    padding = np.zeros(nfft)

    proc = np.concatenate(signal, padding)
    result = np.empty((nsegs, nfft), dtype=np.float32)
    for i in range(nsegs):
        current = proc[offset:offset + nfft]
        curved = current * hanning
        padded = np.append(curved, padding)
        spectrum = np.fft.fft(padded) / nfft


def spectogram(signal, sr):
    pass
