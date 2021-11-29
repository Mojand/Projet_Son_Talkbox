import matplotlib.pyplot as plt
import librosa.display
import os
import numpy as np
from scipy import signal
import IPython.display as ipd
import soundfile as sf
import copy
from LCP import filtre

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

    print("sr="+str(sr))

    plt.figure(1)
    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(audio , sr=sr)
    plt.show()

    return audio, sr 

def read_vocal_audio(audio,sr):
    """Read a vocal audio."""
    ipd.Audio(audio, rate=sr)

def save_vocal_audio(audio,sr):
    """Save a vocal audio."""
    sf.write("audio/pyaudio_output.wav", audio, sr)

def segm_vocal_audio(audio,sr):
    """Segmenter un signal audio en segments de 20ms.

    Args:
        audio (np.ndarray): the audio signal
        sr (float): The sample rate of the audio file

    Returns:
        frames (np.ndarray) : les segments de 20ms du signal audio
        nb_ech_segm : nombre d'echantillons par segment de 20ms

    """

    nb_ech_segm=int(0.02*sr) #Un segment de 20ms correspond a 441 points.
    print("taille_frames="+str(nb_ech_segm)) 
    nb_ech_mix=int(np.ceil(0.4*nb_ech_segm))

    frames=librosa.util.frame(audio,frame_length=nb_ech_segm,hop_length=nb_ech_segm-nb_ech_mix,axis=0) #Le signal est divise en 168 paquets
    #Pour l'overlap prendre au maximum la moitie de la frame (hop_length) par ex : (1-1/4)nb_ech_segm
    print("nb_frames="+str(frames.shape)) 

    return frames,nb_ech_segm, nb_ech_mix



def apply_window(audio,nb_ech_segm):
    """Fenetrage du signal audio.

    Args:
        audio (np.ndarray): the audio signal
        sr (float): The sample rate of the audio file

    Returns:
        audio_window (np.ndarray) : la signal fenetre dans le domaine frequentiel

    """

    plt.figure(2)
    window=signal.windows.hamming(nb_ech_segm)
    plt.plot(window)

    audio_window=[a*w for a,w in zip(audio,window)]

    return audio_window



def fenetre_rampe(nb_ech_segm,nb_ech_mix) :
    """Reconcatener les trames de  20 ms en un signal vocal"""
    fenetre=[]
    for i in range (nb_ech_segm) :
        if i in np.arange(nb_ech_mix):
            fenetre.append(i/(nb_ech_mix-1))
        elif i in range(nb_ech_segm-nb_ech_mix, nb_ech_segm):
            fenetre.append(1-(i-(nb_ech_segm-nb_ech_mix))/(nb_ech_mix-1))
        else :
            fenetre.append(1)
    
    plt.figure(3)
    plt.plot(fenetre)
    plt.show()

    return fenetre

def concatenate(segms_audio, fenetre, nb_ech_mix):

    audios_fenetre=[]
    cpt=0
    for i in segms_audio :
        audios_fenetre.append(np.array(i)*np.array(fenetre))
        audios_fenetre[cpt]=audios_fenetre[cpt].tolist()
        cpt=cpt+1
    plt.figure(4)
    plt.plot(segms_audio[450])
    plt.plot(audios_fenetre[450])
    plt.plot((np.array(fenetre)*max(audios_fenetre[450])).tolist())
    plt.plot((np.array(fenetre)*min(audios_fenetre[450])).tolist())
    plt.show()

    concatenation=audios_fenetre[0] 
    #concatenation.append(audios_fenetre[0].tolist())
    for i in range (1,len(audios_fenetre)) :
        #Concatenation des rampes
        for j in range (nb_ech_mix,1,-1):
            concatenation[-j]=concatenation[-j]+audios_fenetre[i][nb_ech_mix-j] #len(concatenation)
        #Le reste 
        concatenation=concatenation+audios_fenetre[i][nb_ech_mix:]

    plt.figure(5)
    plt.plot(concatenation)
    plt.show()

    return concatenation


##################################################################

#Chargement de l audio
print(os.listdir('audio'))
audio_voix, sr_voix=load_vocal_audio('audio/tempetes.wav')
audio_piano, sr_piano=load_vocal_audio('audio/piano.wav')

#On enlève le blanc au début du son du piano
ind=0
while(audio_piano[ind]<0.01):
    ind=ind+1
audio_piano = audio_piano[ind::]

#Segmentation en segments de 20ms
audio_segm_voix,nb_ech_segm_voix,nb_ech_mix_voix=segm_vocal_audio(audio_voix,sr_voix)
audio_segm_piano,nb_ech_segm_piano,nb_ech_mix_piano=segm_vocal_audio(audio_piano,sr_piano)

#Fenetre de Hamming
audio_window=[]
audio_filtre=[]

if len(audio_segm_voix)>len(audio_segm_piano) :
    minimum=len(audio_segm_piano)
else :
    minimum=len(audio_segm_voix)
for i in range (minimum) :
    audio_window.append(apply_window(audio_segm_voix[i],nb_ech_segm_voix))
    #filtre
    audio_filtre.append(filtre(audio_window[i],audio_segm_piano[i],9))
    

#Calcul de la fenetre rampe
fenetre=fenetre_rampe(nb_ech_segm_piano,nb_ech_mix_piano)

#Concatenation des trames de 20ms
audio_conc=concatenate(audio_filtre, fenetre, nb_ech_mix_piano)

plt.figure(6)
plt.plot(audio_piano)
plt.show()

#Enregistrement du resultat obtenu
save_vocal_audio(audio_conc,sr_piano)

