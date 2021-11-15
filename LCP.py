import numpy as np

def autoCorrI(signal, i):
    
    sum = 0
    size = len(signal)

    i = np.absolute(i)

    for j in range(size):

        sum += signal[j]*signal[i+j]
    return sum;


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

    return np.linalg.inv(matriceR(signal = signal,i = p))

def Durbin(vectR, p, R0):

    rho = R0
    A0 = np.zeros(0)

    for i in range (1,p+1):
        vectTemp = vectR[0:(p-1)][:: -1]
        
        kp = (vectR(i) + np.dot(A0,vectTemp))/rho

        A0 = np.concatenate((A0,np.zeros(1))) - kp*(  np.concatenate((  np.dot(np.eye(i-1)[:: -1],A0),   np.ones(1)  ))  )

        rho = (1 - kp*kp)*rho 


    return A0

def LCP(signal, p):

    vectR = vecteurR(signal,p)
    R0 = autoCorrI(signal,0)
    A = Durbin(vectR=vectR,p = p,R0 = R0)
    
    return A
