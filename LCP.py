import numpy as np
#from traitement_audio import *

def autoCorrI(signal, i):
    
    sum = 0
    size = len(signal)

    i = np.absolute(i)

    for j in range(size)-i:
        sum += signal[j]*signal[i+j]
    return sum


def matriceR(signal, p):

    mat = np.zeros((p,p))

    for i in range (p):
        for j in range(i,p):

            mat[i,j] = autoCorrI(signal=signal, i= (j-i))
            if (i != j):
                mat[j,i]=mat[i,j]

    return mat

def vecteurR(signal, p):

    vect = np.zeros(p)

    for i in range(1, p+1):
        vect[i-1] = autoCorrI(signal=signal,i = i)

    return vect 

def matriceRInv(signal,p):
    R0 = autoCorrI(signal,0)
    if (R0!=0) :
        return np.linalg.inv(matriceR(signal = signal,p = p))
    else :
        return np.zeros((p,p))

def Durbin(vectR, p, R0):

    rho = R0
    A0 = np.zeros(p)
    vectTemp = np.asarray(vectR)

    for i in range (0,p):
        #print("i = ",i)
        #print(vectR)
        #print(A0)
        #print(vectTemp)
        #print(np.dot(A0,vectTemp))

        dot = np.dot(A0[0:i],vectTemp[0:i][:: -1])
        #print(dot)


        kp = (vectR[i] + dot)/rho
        #print(A0)
        #A0 = np.concatenate((A0,np.zeros(1))) - kp*(  np.concatenate((  np.dot(np.eye(i-1)[:: -1],A0),   np.ones(1)  ))  )

        #print(A0[0:i])
        #print(vectTemp[0:i])

        A0[0:i] = A0[0:i] - kp*A0[0:i][:: -1]
        A0[i] = - kp
        #print("rho = ",rho)
        rho = (1 - kp*kp)*rho 
        
    #print(vectR)

    return A0

def LCP(signal, p):

    vectR = vecteurR(signal,p)
    R0 = autoCorrI(signal,0)
    #print(R0)
    if (R0!=0) :
        A = Durbin(vectR=vectR,p = p,R0 = R0)
    else :
        A=np.zeros((p,1))
    
    return A

def filtre(voix, instrument, p):

    estime = np.zeros(len(instrument))
    #A0 = LCP(signal=voix,p=p)
    A0=np.dot(matriceRInv(signal=voix,p=p),vecteurR(signal=voix,p=p))
    for i in range(len(instrument)):
        for j in range(1,p):
            if (i - j) > 0:
                estime[i] += instrument[i-j]*A0[j]
    return estime

#if __name__ == '__main__':

    # #Chargement de l'audio
    # audio, sr=load_vocal_audio('audio/voix.wav')

    # #Segmentation en segments de 20ms
    # audio_segm,nb_ech_segm=segm_vocal_audio(audio,sr)

    # #Fenetre de Hamming
    # audio_window=[]
    # for i in range (len(audio_segm)) :
    #     audio_window.append(apply_window(audio_segm[i],nb_ech_segm))

    # #Concatenation des trames de 20ms
    # audio_conc=concatenate(audio_window)

    # #Enregistrement du résultat obtenu
    # save_vocal_audio(audio_conc,sr)

    # #print(len(audio_window))
    # #print(len(audio_segm[1]))
    # #print(vecteurR(audio_segm[1],12))
    
    # print(np.dot(matriceRInv(audio_segm[1],12),vecteurR(audio_segm[1],12)))

    # print(LCP(audio_segm[1],12))