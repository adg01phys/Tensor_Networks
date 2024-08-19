#Basic Manipulations of matrix product states. 
'''References: 
https://github.com/GCatarina/DMRG_MPS_didactic/blob/main/DMRG-MPS_implementation.ipynb
arXiv:1805.00055
arXiv:2304.13395
'''
#Constructs Random Matrix Product States
#Converts mps to Left/Right Canonical Forms using reshape and svd
#uses einsum from numpy for tensor contractions
import numpy as np

def Random_MPS(N,d,D):
    Mrand=[]
    Mrand.append(np.random.rand(1,d,D))         #tensor at left edge
    for l in range(1,N-1):
        Mrand.append(np.random.rand(D,d,D))     #tensors in the bulk
    Mrand.append(np.random.rand(D,d,1))         #tensor at right edge
    return Mrand

#The canonical forms make use of the singular value decomposition to split tensors at each site. The left/right bond indices are bundled with the physical index and the resulting matrix is decomposed using an SVD.
def LeftCan(M):             #SVD: U, SV_dag
    Mcopy = M.copy()
    N = len(Mcopy)
    for l in range(N):
        Ml = Mcopy[l]   #pick out tensor at site l
        Ml = np.reshape(Ml,(np.shape(Ml)[0]*np.shape(Ml)[1],np.shape(Ml)[2])) # bundle physical index with right bond-index
        U,S,V_dag = np.linalg.svd(Ml,full_matrices=False)
        Mcopy[l] = np.reshape(U,(np.shape(Mcopy[l])[0],np.shape(Mcopy[l])[1],np.shape(U)[1])) #left unitary goes to site l
        SV_dag = np.matmul(np.diag(S),V_dag)
        if l<N-1:
            Mcopy[l+1] = np.einsum('ij,jkl',SV_dag, Mcopy[l+1]) #Singular values and right unitary go to site l+1
    return Mcopy
    
def RightCan(M):             #SVD: US, V_dag
    Mcopy = M.copy()
    N = len(Mcopy)
    for l in range(N-1,-1,-1):
        Ml = Mcopy[l]   #pick out tensor at site l
        Ml = np.reshape(Ml,(np.shape(Ml)[0],np.shape(Ml)[1]*np.shape(Ml)[2])) # bundle physical index with left bond-index
        U,S,V_dag = np.linalg.svd(Ml,full_matrices=False)
        Mcopy[l] = np.reshape(V_dag,(np.shape(V_dag)[0],np.shape(Mcopy[l])[1],np.shape(Mcopy[l])[2])) #right unitary goes to site l
        US = np.matmul(U,np.diag(S))
        if l>0:
            Mcopy[l-1] = np.einsum('ijk,kl',Mcopy[l-1],US) #Singular values and left unitary go to site l-1
    return Mcopy

N=10
d=2
D=3
M_0 = Random_MPS(N,d,D)

Mleft = LeftCan(M_0)
Mright = RightCan(M_0)

#Display dimensions of the MPS
print("Dimensions of random MPS")
for l in range(N):
    print('l =', l, ':', np.shape(M_0[l]))
print("Dimensions of MPS in Left Canonical Form")
for l in range(N):
    print('l =', l, ':', np.shape(Mleft[l]))
print("Dimensions of MPS in Right Canonical Form")
for l in range(N):
    print('l =', l, ':', np.shape(Mright[l]))
    
#Check for Normalisation
#LeftCan

for l in range(N):
    Mdag = Mleft[l].conj().T 
    MdagM = np.einsum('ijk,kjl',Mdag,Mleft[l])  #the order of the indices is reversed on conjugation
    I = np.eye(np.shape(Mleft[l])[2])
    print('l =', l, ': max(|M_l[l]^† · M_l[l] - I|) =', np.max(abs(MdagM-I)))           #check normalisation. Should be close to zero

for l in range(N):
    Mdag = Mright[l].conj().T 
    MdagM = np.einsum('ijk,kjl',Mright[l],Mdag)  #the order of the indices is reversed on conjugation
    I = np.eye(np.shape(Mright[l])[0])
    print('l =', l, ': max(|M_r[l]^† · M_r[l] - I|) =', np.max(abs(MdagM-I)))           #check normalisation. Should be close to zero




