
import matplotlib.pyplot as plt
import librosa.display
import os
import numpy as np
from scipy import signal
import IPython.display as ipd
import soundfile as sf

def load_vocal_audio(audio_path):
    """Load a vocal audio.

    Args:
        audio_path (str): path to audio file

    Returns:
        audio (np.ndarray): the audio signal
        sr (float): The sample rate of the audio file

    """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))

    audio, sr = librosa.load(audio_path)

    #print("sr="+str(sr))

    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(audio , sr=sr)
    #plt.show()

    return audio, sr 

def read_vocal_audio(audio,sr):
    """Read a vocal audio."""
    ipd.Audio(audio, rate=sr)

def save_vocal_audio(audio,sr):
    """Save a vocal audio."""
    sf.write("audio/pyaudio_output.wav", audio, sr)

def segm_vocal_audio(audio,sr):
    """Ségmenter un signal audio en segments de 20ms.

    Args:
        audio (np.ndarray): the audio signal
        sr (float): The sample rate of the audio file

    Returns:
        frames (np.ndarray) : les segments de 20ms du signal audio
        nb_ech_segm : nombre d'echantillons par segment de 20ms

    """

    nb_ech_segm=int(0.02*sr) #Un segment de 20ms correspond à 441 points.
    #print("taille_frames="+str(nb_ech_segm)) 

    frames=librosa.util.frame(audio,frame_length=nb_ech_segm,hop_length=nb_ech_segm,axis=0) #Le signal est divisé en 168 paquets
    #print("nb_frames="+str(frames.shape)) 

    return frames,nb_ech_segm



def apply_window(audio,nb_ech_segm):
    """Fenetrage du signal audio.

    Args:
        audio (np.ndarray): the audio signal
        sr (float): The sample rate of the audio file

    Returns:
        audio_window (np.ndarray) : la signal fenetré dans le domaine fréquentiel

    """

    window=signal.windows.hamming(nb_ech_segm)
    plt.plot(window)

    audio_window=[a*w for a,w in zip(audio,window)]

    return audio_window



def concatenate(segm_audio) :
    """Reconcaténer les trames de  20 ms en un signal vocal"""
    return np.concatenate(segm_audio)


##################################################################

#Chargement de l'audio
audio, sr=load_vocal_audio('audio/voix.wav')

#Segmentation en segments de 20ms
audio_segm,nb_ech_segm=segm_vocal_audio(audio,sr)

#Fenetre de Hamming
audio_window=[]
for i in range (len(audio_segm)) :
    audio_window.append(apply_window(audio_segm[i],nb_ech_segm))

#Concatenation des trames de 20ms
audio_conc=concatenate(audio_window)

#Enregistrement du résultat obtenu
save_vocal_audio(audio_conc,sr)

