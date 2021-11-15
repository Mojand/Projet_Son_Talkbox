#Nicolas Obin

import matplotlib.pyplot as plt
import librosa.display
import os
from scipy import signal

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

    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(audio , sr=sr)
    plt.show()

    return audio, sr 

def segm_vocal_audio(audio,sr):
    """Ségmenter un signal audio en segments de 20ms.

    Args:
        audio (np.ndarray): the audio signal
        sr (float): The sample rate of the audio file

    Returns:
        frames (np.ndarray) : les segments de 20ms du signal audio

    """

    segm_length=int(0.02*sr) #Un segment de 20ms correspond à 441 points.
    print("taille_frames="+str(segm_length)) 

    frames=librosa.util.frame(audio,frame_length=segm_length,hop_length=segm_length,axis=0) #Le signal est divisé en 168 paquets
    print("nb_frames="+str(frames.shape)) 

    return frames


def visualize_vocal_audio_spec(X,sr):
    """Répresentation du spectrogramme.

    Args:
        X (np.ndarray): le signal audio en fréquentiel
        sr (float): The sample rate of the audio file

    """
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.show()



def apply_window(audio,sr):
    """Fenetrage du signal audio.

    Args:
        audio (np.ndarray): the audio signal
        sr (float): The sample rate of the audio file

    Returns:
        X (np.ndarray) : la signal fenetré dans le domaine fréquentiel

    """

    window=signal.windows.hamming(441)
    plt.plot(window)

    X_avant = librosa.stft(audio,n_fft=441)
    visualize_vocal_audio_spec(X_avant,sr)

    X = librosa.stft(audio, window=window, n_fft=441)
    visualize_vocal_audio_spec(X,sr)

    return X

def reverse_temporal(audio_freq):
    audio=librosa.istft(audio_freq)


audio, sr=load_vocal_audio('audio/voix.wav')
audio_segm=segm_vocal_audio(audio,sr)
apply_window(audio_segm[100],sr)

