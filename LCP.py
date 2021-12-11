import numpy as np
from scipy.signal import lfilter, filtfilt


def autoCorrI(signal, i):
    """Calcul d'autocorrélation à l'indice i

    Args:
        signal: le signal
        i :     l'indice

    Return:
        somme : l'autocorrélation du signal à l'indice i : R_i

    """
    
    somme = 0
    size = len(signal)
    i = np.absolute(i)

    for j in range(size)-i:
        somme += signal[j]*signal[i+j]
    return somme


def matriceR(signal, p):
    """ Implémentation de la matrice symétrique R 
    Args:
        signal: le signal
        p :     l'ordre du filtre => dimensions de la matrice

    Return:
        mat : la matrice R
    """

    mat = np.zeros((p,p))

    for i in range (p):
        for j in range(i,p):
            mat[i,j] = autoCorrI(signal=signal, i= (j-i))
            if (i != j):
                mat[j,i]=mat[i,j]

    return mat

def vecteurR(signal, p):
    """ calcul du vecteur R = [R1 R2 ... Rp]
    Args:
        signal: le signal
        p :     l'ordre du filtre => dimensions de la matrice

    Return:
        A : les coeeficients du filtre
    """
    vect = np.zeros(p)

    for i in range(1, p+1):
        vect[i-1] = autoCorrI(signal=signal,i = i)

    return vect 

def matriceRInv(signal,p):
    """ calcul de l'inverse de la matrice R (résolution)
    Args:
        signal: le signal
        p :     l'ordre du filtre => dimensions de la matrice

    Return:
        mat : l'inverse de la matrice R
    """
    R0 = autoCorrI(signal,0)
    if (R0!=0) :
        return np.linalg.inv(matriceR(signal = signal,p = p))
    else :
        return np.zeros((p,p))

def Durbin(vectR, p, R0):
    """ Algorithme de Durbin-Levinson : diminution de la complexité
    de l'algorithme
    Args:
        vectR: le vecteur [R1 R2 .. Rp]
        p :     l'ordre du filtre => dimensions de la matrice
        R0 : la valeur de l'auto-corrélation R0

    Return:
        A0 : les coefficients du filtre
    """
    rho = R0
    A0 = np.zeros(p)
    vectTemp = np.asarray(vectR)

    for i in range (0,p):

        dot = np.dot(A0[0:i],vectTemp[0:i][:: -1])
        kp = (vectR[i] + dot)/rho
        A0[0:i] = A0[0:i] - kp*A0[0:i][:: -1]
        A0[i] = - kp
        rho = (1 - kp*kp)*rho 
        
    return A0

def LPC(signal, p, methode):
    """ Algorithme de détermination des coefficient du filtre 
    du signal de voix signal par la méthode de Durbin-Levinson
    Args:
        signal: le signal
        p :     l'ordre du filtre => dimensions de la matrice
        methode : chaine de caractères qui correspond à la méthode
        à employer : Durbin-Levinson ou l'inversion de la matrice R


    Return:
        A : les coeeficients du filtre
    """

    vectR = vecteurR(signal,p)
    R0 = autoCorrI(signal,0)
    # si R0=0 alors la trame signal ne possède pas de parole
    # on ne recherche alors pas les coefficients du filtre : R
    # n'est pas inversible
    if (R0!=0) :
        if methode == "Durbin" :
            A = Durbin(vectR=vectR,p = p,R0 = R0)
        else : #methode = Rinverse
            A = - np.dot(matriceRInv(signal=signal,p=p),vecteurR(signal=signal,p=p))
    else :
        A=np.zeros(p)
    
    return A

def filtre(voix, instrument, p, methode):
    """ Fonction finale : cette fonction détermine le filtre du 
    locuteur sur une trame voix et applique le filtre sur la trame
    instrument
    Args:
        voix:       le signal de parole
        instrument: le signal de l'instrument qui sera filtré
        p :         l'ordre du filtre => dimensions de la matrice
        methode:    chaine de caractères qui correspond à la méthode
        à employer, soit Durbin-Levinson soit l'inversion de la matrice R


    Return:
        estime : le signal de l'instrument filtré
    """
    estime = np.zeros(len(instrument))
    
    A0 = LPC(signal=voix,p=p, methode=methode)
    
    a = np.hstack([[1], 1 * A0])
   
    estime = lfilter([1], a, instrument)
    #estime = filtfilt([1], a, instrument)

    return estime

