import matplotlib.pyplot as plt
import librosa.display
import os
import numpy as np
from scipy import signal
import IPython.display as ipd
import soundfile as sf
import copy
from LPC import filtre
import argparse

def load_vocal_audio(audio_path):
    """Charger un fichier audio """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))
    audio, sr = librosa.load(audio_path)
    
    return audio, sr 

def read_vocal_audio(audio,sr):
    """Lire un fichier audio"""
    ipd.Audio(audio, rate=sr)

def save_vocal_audio(audio,sr):
    """Sauvegarder un fichier audio"""
    sf.write("audio/pyaudio_output.wav", audio, sr)

def segm_vocal_audio(audio,sr):
    """Segmenter un signal audio en segments de 20ms."""

    nb_ech_segm=int(0.02*sr) #Un segment de 20ms correspond a 441 points.
    nb_ech_mix=int(np.ceil(0.25*nb_ech_segm))

    frames=librosa.util.frame(audio,frame_length=nb_ech_segm,hop_length=nb_ech_segm-nb_ech_mix,axis=0) #Le signal est divise en 168 paquets
    #Pour l'overlap prendre au maximum la moitie de la frame (hop_length) par ex : (1-1/4)nb_ech_segm

    return frames,nb_ech_segm, nb_ech_mix



def apply_window(audio,nb_ech_segm):
    """Application de la fenetre de la fenetre de Hamming"""

    window=signal.windows.hamming(nb_ech_segm)

    audio_window = np.zeros((len(window),1))

    for i in range(len(window)):
        audio_window[i] = audio[i]*window[i]

    return audio_window



def fenetre_rampe(nb_ech_segm,nb_ech_mix, display) :
    """Modélisation de la rampe pour le concatenation des trames"""
    fenetre=[]
    for i in range (nb_ech_segm) :
        if i in np.arange(nb_ech_mix):
            fenetre.append(i/(nb_ech_mix-1))
        elif i in range(nb_ech_segm-nb_ech_mix, nb_ech_segm):
            fenetre.append(1-(i-(nb_ech_segm-nb_ech_mix))/(nb_ech_mix-1))
        else :
            fenetre.append(1)
    if display == "True" :
        plt.figure(3)
        plt.plot(fenetre)
        plt.xlabel('Echantillons')
        plt.ylabel('Gain')
        plt.title("Rampe")
        plt.show()

    return fenetre

def concatenate(segms_audio, fenetre, nb_ech_mix, display):
    """Reconcatener les trames de  20 ms en un signal vocal"""
    audios_fenetre=[]
    cpt=0
    for i in segms_audio :
        audios_fenetre.append(np.array(i)*np.array(fenetre))
        audios_fenetre[cpt]=audios_fenetre[cpt].tolist()
        cpt=cpt+1

    print()
    
    if display=="True" :
        plt.figure(4)
        plt.plot(segms_audio[int(len(segms_audio)/2)])
        plt.plot(audios_fenetre[int(len(segms_audio)/2)])
        plt.plot((np.array(fenetre)*max(audios_fenetre[int(len(segms_audio)/2)])).tolist())
        plt.plot((np.array(fenetre)*min(audios_fenetre[int(len(segms_audio)/2)])).tolist())
        plt.xlabel('Echantillons')
        plt.ylabel('Amplitude')
        plt.title('Trame soumise à une rampe pour la concaténation')
        plt.show()
    
    concatenation=audios_fenetre[0] 
    for i in range (1,len(audios_fenetre)) :
        #Concatenation des rampes
        for j in range (nb_ech_mix,1,-1):
            concatenation[-j]=concatenation[-j]+audios_fenetre[i][nb_ech_mix-j] 
        #Le reste 
        concatenation=concatenation+audios_fenetre[i][nb_ech_mix:]

    if display=="True" :
        plt.figure(5)
        plt.plot(concatenation)
        plt.title("Signal de sortie")
        plt.show()

    return concatenation


##################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser( usage=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-methode', nargs='?', type=str, default='Durbin',  help='methode de determination des coefficients : Durbin ou Rinverse')
    parser.add_argument('-audio', nargs='?', type=str, default='audio/tempetes.wav',  help='chemin du fichier de parole à partir du répertoire courant')
    parser.add_argument('-instrument', nargs='?', type=str, default="audio/bruit_blanc.wav", help='chemin du fichier d instrument à partir du répertoire courant')
    parser.add_argument('-ordre', nargs='?', type=int, default=10,  help='ordre du filtre')
    parser.add_argument('-derive', nargs='?', type=str, default="True",  help='ajout d un filtre de pre-accentuation (dérivateur) si derive = True')
    parser.add_argument('-display', nargs='?', type=str, default="True",  help='affichage des graphiques si display=True')
    args = parser.parse_args()

    #Chargement de l audio
    print("Chargement des audios")
    audio_voix, sr_voix=load_vocal_audio(args.audio)
    if args.derive == "True" :
        for i in range(1,len(audio_voix)) :
            audio_voix[i] = audio_voix[i] - audio_voix[i-1]
    audio_piano, sr_piano=load_vocal_audio(args.instrument)

    #On enlève le blanc au début du son du piano
    ind=0
    while(audio_piano[ind]<0.01):
        ind=ind+1
    audio_piano = audio_piano[ind::]


    #Segmentation en segments de 20ms
    print("Segmentation des signaux audio en trames de 20 ms")
    audio_segm_voix,nb_ech_segm_voix,nb_ech_mix_voix=segm_vocal_audio(audio_voix,sr_voix)
    audio_segm_piano,nb_ech_segm_piano,nb_ech_mix_piano=segm_vocal_audio(audio_piano,sr_piano)

    
    audio_window=[]
    audio_filtre=[]
    audio_filtre2=[]


    if len(audio_segm_voix)>len(audio_segm_piano) :
        minimum=len(audio_segm_piano)
    else :
        minimum=len(audio_segm_voix)


    print("Filtrage LCP")
    for i in range (minimum) :
        #Fenetre de Hamming
        audio_window.append(apply_window(audio_segm_voix[i],nb_ech_segm_voix))
        #Filtrage LCP
        audio_filtre.append(filtre(audio_window[i],audio_segm_piano[i],args.ordre,args.methode))

    #Calcul de la fenetre rampe
    fenetre=fenetre_rampe(nb_ech_segm_piano,nb_ech_mix_piano, args.display)

    #Concatenation des trames de 20ms
    print("Concatenation des trames obtenues par filtrage LCP")
    audio_conc=concatenate(audio_filtre, fenetre, nb_ech_mix_piano, args.display)

    audio_conc = np.array(audio_conc)

    # Comparaison des spectres
    specVoix = np.abs(librosa.stft(audio_voix[sr_voix:sr_voix*5]))
    specInstru = np.abs(librosa.stft(audio_piano[sr_piano:sr_piano*5]))
    specFiltre = np.abs(librosa.stft(audio_conc[sr_piano:sr_piano*5]))

    if args.display == "True" :
        fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
        librosa.display.specshow(librosa.amplitude_to_db(specVoix, ref=np.max),y_axis='log', x_axis='time', ax=ax[0])
        plt.title('spectre de la voix')
        librosa.display.specshow(librosa.amplitude_to_db(specInstru, ref=np.max),y_axis='log', x_axis='time', ax=ax[1])
        plt.title('spectre de l instrument')
        librosa.display.specshow(librosa.amplitude_to_db(specFiltre, ref=np.max),y_axis='log', x_axis='time', ax=ax[2])
        plt.title('spectre de l instrument filtré')
        plt.show()

    #Enregistrement du resultat obtenu
    print("Enregistrement du signal obtenu : signal modelisant le TalkBox")
    save_vocal_audio(audio_conc,sr_piano)

