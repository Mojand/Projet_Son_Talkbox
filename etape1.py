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
from LPC import filtre
import time

from traitement_audio import apply_window

audio_bruit, sr1=load('audio/bruit_blanc.wav')

# on enlève les zeros sur le signal de parole
y, sr = librosa.load('audio/A.wav', duration=0.020, offset=1, sr=sr2)

y = np.squeeze(apply_window(y,len(y)))

#recherche des paramètres du filtre sur ce signal puis on applique le filtre sur le bruit
start_time = time.time()
bruit_filtre = filtre(audio_voix,audio_bruit,10,"Durbin") #"Durbin" ou "Rinverse"
print("--- %s seconds ---" % (time.time() - start_time))

plt.figure(1)
plt.plot(bruit_filtre)
plt.plot(audio_bruit)
plt.show()

specVoix = np.abs(librosa.stft(audio_voix[sr2:sr2*5]))
specBruit = np.abs(librosa.stft(audio_bruit[sr1:sr1*5]))
specFiltre = np.abs(librosa.stft(bruit_filtre[sr1:sr1*5]))

fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
librosa.display.specshow(librosa.amplitude_to_db(specVoix, ref=np.max),y_axis='log', x_axis='time', ax=ax[0])

librosa.display.specshow(librosa.amplitude_to_db(specBruit, ref=np.max),y_axis='log', x_axis='time', ax=ax[1])

librosa.display.specshow(librosa.amplitude_to_db(specFiltre, ref=np.max),y_axis='log', x_axis='time', ax=ax[2])
plt.show()

sf.write("audio/pyaudio_output.wav", bruit_filtre, sr1)
