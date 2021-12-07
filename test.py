import matplotlib.pyplot as plt
import librosa.display
from librosa import load
from librosa import lpc
from scipy.signal import lfilter
import os
import numpy as np
from scipy import signal
import IPython.display as ipd
import soundfile as sf
import copy
from LCP import filtre

audio_bruit, sr1=load('audio/bruit_blanc.wav')
audio_voix, sr2=load('audio/AEIOU.wav')

# on enlève les zeros
print((len(audio_voix)/2)/sr2)
y, sr = librosa.load('audio/A.wav', duration=0.020, offset=1, sr=sr2)

#recherche des paramètres du filtre sur ce signal
A0 = lpc(y, 50)
print("coeff filtre : ",A0)

a = np.hstack([[1], 1 * A0[1:]])
print("a = ", a)
bruit_filtre = lfilter([1], a, audio_bruit)

plt.figure(1)
plt.plot(bruit_filtre)
plt.plot(audio_bruit)
plt.show()

specVoix = np.abs(librosa.stft(audio_voix[sr2:sr2*5]))
specBruit = np.abs(librosa.stft(audio_bruit[sr1:sr1*5]))
specFiltre = np.abs(librosa.stft(bruit_filtre[sr1:sr1*5]))

plt.figure(2)
fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
librosa.display.specshow(librosa.amplitude_to_db(specVoix, ref=np.max),y_axis='log', x_axis='time', ax=ax[0])

librosa.display.specshow(librosa.amplitude_to_db(specBruit, ref=np.max),y_axis='log', x_axis='time', ax=ax[1])

librosa.display.specshow(librosa.amplitude_to_db(specFiltre, ref=np.max),y_axis='log', x_axis='time', ax=ax[2])
plt.show()

sf.write("audio/pyaudio_output.wav", bruit_filtre, sr1)
