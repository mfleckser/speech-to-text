import numpy as np
import tensorflow as tf
import wave
import preprocessing

audio, samples, sr = preprocessing.load("samples/simons4.wav")
print(samples)
preprocessing.spectogram(audio, samples, sr)
