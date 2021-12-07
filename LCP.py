import numpy as np
from librosa import lpc
from scipy.signal import lfilter


def autoCorrI(signal, i):
    """Calcul d'autocorrélation à l'indice i

    Args:
        signal: le signal
        i :     l'indice

    Return:
        somme : l'autocorrélation du signal à l'indice i

    """
    
    somme = 0
    size = len(signal)

    i = np.absolute(i)

    for j in range(size)-i:
        somme += signal[j]*signal[i+j]
    return somme


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

        dot = np.dot(A0[0:i],vectTemp[0:i][:: -1])
        #print(dot)


        kp = (vectR[i] + dot)/rho
        
        A0[0:i] = A0[0:i] - kp*A0[0:i][:: -1]
        A0[i] = - kp

        rho = (1 - kp*kp)*rho 
        

    return A0

def LCP(signal, p):

    vectR = vecteurR(signal,p)
    R0 = autoCorrI(signal,0)
    #print(R0)
    if (R0!=0) :
        A = Durbin(vectR=vectR,p = p,R0 = R0)
    else :
        A=np.zeros(p)
    print(A.shape)
    
    return A

def filtre(voix, instrument, p):

    estime = np.zeros(len(instrument))
    
    A0 = LCP(signal=voix,p=p)
    
    #A0_2=np.dot(matriceRInv(signal=voix,p=p),vecteurR(signal=voix,p=p))
    
    """
    LPC LIBROSA (pour tests)
    R0 = autoCorrI(voix,0)
    if R0 != 0 :
        A0 = lpc(voix, p)
        #A0 = A0[1:]
    else :
        A0=np.zeros(p)
    
    print("librosa : ", A0)
    """

    #a = np.hstack([[1], 1 * A0[1:]]) # pour LPC librosa
    a = np.hstack([[1], 1 * A0])
    print("a = ", a)
    estime = lfilter([1], a, instrument)

    return estime

