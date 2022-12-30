from math import floor
import wave
import math
import numpy as np
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
    nfft = 256
    hop_size = 128
    nsegs = math.floor(len(signal) / hop_size)

    hanning = np.hanning(nfft)
    padding = np.zeros(nfft, dtype=np.float32)

    proc = np.concatenate((signal, padding))
    result = np.empty((nsegs, nfft), dtype=np.float32)
    for i in range(nsegs):
        current = proc[offset:offset + nfft]
        offset += hop_size
        curved = current * hanning
        padded = np.append(curved, padding)
        spectrum = np.fft.fft(padded) / nfft
        autopower = np.abs(spectrum * np.conj(spectrum))
        result[i, :] = autopower[:nfft]

    result = 20*np.log10(result)  # convert to dB
    result = np.clip(result, -40, 200)

    return result.transpose()


def spectrogram(signal, sr):
    #fig, (ax1, ax2) = plt.subplots(nrows=2)

    amplitudes = stft(signal, sr)
    # plt.axes(yscale="log")
    plt.imshow(amplitudes, origin="lower", cmap="jet",
               interpolation="nearest", aspect="auto")
    #spectrum, freqs, t, im = plt.specgram(signal, Fs=sr, mode="magnitude")
    # amps = 20*np.log10(np.square(spectrum))  # dB conversion
    #amps = np.clip(amps, -40, 200)
    # plt.xticks(t)
    # plt.yticks(freqs)
    # ax2.imshow(amps, origin="lower", aspect="auto",
    #           interpolation="nearest", cmap="jet")
    plt.show()
    #print(f"My Shape: ${amplitudes.shape}, PLT Shape: ${amps.shape}")
