import numpy as np
import tensorflow as tf
import wave
import preprocessing

audio, sr = preprocessing.load("samples/simons4.wav")
preprocessing.spectrogram(audio, sr)
