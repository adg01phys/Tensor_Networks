#1-site dmrg for finite-size transverse-field Ising Model using Matrix Product States
'''References: 
    https://github.com/GCatarina/DMRG_MPS_didactic/blob/main/DMRG-MPS_implementation.ipynb
   	arXiv:2304.13395
    arXiv:1805.00055
    https://mcgreevy.physics.ucsd.edu/s14/index.html
'''

import numpy as np
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt 

sp = np.zeros((2,2))
sp[0,1] = 1
sm = np.zeros((2,2))
sm[1,0] = 1
sx = np.zeros((2,2))
sx[0,1]=1
sx[1,0]=1
I2 = np.eye(2)
sz = np.zeros((2,2))
sz[0,0] = 1
sz[1,1] = -1

def Random_MPS(N,d,D):
    Mrand=[]
    Mrand.append(np.random.rand(1,d,D))         #tensor at left edge
    for l in range(1,N-1):
        Mrand.append(np.random.rand(D,d,D))     #tensors in the bulk
    Mrand.append(np.random.rand(D,d,1))         #tensor at right edge
    return Mrand
def LeftCan(M):             #SVD: U, SV_dag
    Mcopy = M.copy()
    N = len(Mcopy)
    for l in range(N):
        Ml = Mcopy[l]   
        Ml = np.reshape(Ml,(np.shape(Ml)[0]*np.shape(Ml)[1],np.shape(Ml)[2]))
        U,S,V_dag = np.linalg.svd(Ml,full_matrices=False)
        Mcopy[l] = np.reshape(U,(np.shape(Mcopy[l])[0],np.shape(Mcopy[l])[1],np.shape(U)[1])) 
        SV_dag = np.matmul(np.diag(S),V_dag)
        if l<N-1:
            Mcopy[l+1] = np.einsum('ij,jkl',SV_dag, Mcopy[l+1]) 
    return Mcopy

def RightCan(M):             #SVD: US, V_dag
    Mcopy = M.copy()
    N = len(Mcopy)
    for l in range(N-1,-1,-1):
        Ml = Mcopy[l]   
        Ml = np.reshape(Ml,(np.shape(Ml)[0],np.shape(Ml)[1]*np.shape(Ml)[2])) 
        U,S,V_dag = np.linalg.svd(Ml,full_matrices=False)
        Mcopy[l] = np.reshape(V_dag,(np.shape(V_dag)[0],np.shape(Mcopy[l])[1],np.shape(Mcopy[l])[2]))
        US = np.matmul(U,np.diag(S))
        if l>0:
            Mcopy[l-1] = np.einsum('ijk,kl',Mcopy[l-1],US) 
    return Mcopy

def ZipLeft(Tl,Mb,O,Mt):        #Zipper for MPS contractions 
    #The left tensor Tl stores the result of previous contractions
    Tzip = np.einsum('ijk,klm',Mb,Tl)       #Contract bottom tensor with left tensor
    Tzip = np.einsum('ijkl,kjmn',Tzip,O)    #Contract the resulting tensor with the operator from the MPO
    Tf = np.einsum('ijkl,jlm',Tzip,Mt)      #Contract the result with the top tensor
    return Tf

def ZipRight(Tr,Mb,O,Mt): 
    Tzip = np.einsum('ijk,klm',Mt,Tr)
    Tzip = np.einsum('ijkl,mnkj',Tzip,O) 
    Tf = np.einsum('ijkl,jlm',Tzip,Mb)
    return Tf

def dmrg_1site_obc(H,D,Nsweeps):
    N = len(H)        #no. of sites
    d = np.shape(H[N-1])[3]   #physical dimension
    M = Random_MPS(N,d,D)
    M = LeftCan(M)
    M = RightCan(M)

    Hzip = [np.ones((1,1,1)) for it in range(N+2)]
    for l in range(N-1,-1,-1):
        Hzip[l+1] = ZipRight(Hzip[l+2],M[l].conj().T,H[l],M[l]) #creates left/right environments for each site. H[l] is the local operator from the MPO. 
    Ens=[]    
    for x in range(Nsweeps):
        #right sweep
        for l in range(N):
            Taux = np.einsum('ijk,jlmn',Hzip[l],H[l]) 
            Taux = np.einsum('ijklm,nlo',Taux,Hzip[l+2])
            Taux = np.transpose(Taux,(0,2,5,1,3,4))
            Hmat = np.reshape(Taux,(np.shape(Taux)[0]*np.shape(Taux)[1]*np.shape(Taux)[2],np.shape(Taux)[3]*np.shape(Taux)[4]*np.shape(Taux)[5])) #local effective Hamiltonian 
            val,vec = eigsh(Hmat, k=1, which='SA', v0=M[l])
            Ens.append(val[0])
            #Now update the mps using the local wavefunction:
            Taux2 = np.reshape(vec,(np.shape(Taux)[0]*np.shape(Taux)[1],np.shape(Taux)[2]))
            U,S,V_dag = np.linalg.svd(Taux2,full_matrices=False)
            M[l] = np.reshape(U,(np.shape(Taux)[0],np.shape(Taux)[1],np.shape(U)[1]))
            SV_dag = np.matmul(np.diag(S),V_dag)
            if l < N-1:
                M[l+1] = np.einsum('ij,jkl',SV_dag,M[l+1])
            Hzip[l+1] = ZipLeft(Hzip[l],M[l].conj().T,H[l],M[l])
        #left sweep 
        for l in range(N-1,-1,-1):
            Taux = np.einsum('ijk,jlmn',Hzip[l],H[l])
            Taux = np.einsum('ijklm,nlo',Taux,Hzip[l+2])
            Taux = np.transpose(Taux,(0,2,5,1,3,4))
            Hmat = np.reshape(Taux,(np.shape(Taux)[0]*np.shape(Taux)[1]*np.shape(Taux)[2],np.shape(Taux)[3]*np.shape(Taux)[4]*np.shape(Taux)[5]))
            val,vec = eigsh(Hmat, k=1, which='SA', v0=M[l])
            Ens.append(val[0])
            Taux2 = np.reshape(vec,(np.shape(Taux)[0],np.shape(Taux)[1]*np.shape(Taux)[2]))
            U,S,V_dag = np.linalg.svd(Taux2,full_matrices=False)
            M[l] = np.reshape(V_dag,(np.shape(V_dag)[0],np.shape(M[l])[1],np.shape(M[l])[2]))
            US = np.matmul(U,np.diag(S))
            if l>0:
                M[l-1] = np.einsum('ijk,kl',M[l-1],US)
            Hzip[l+1] = ZipRight(Hzip[l+2],M[l].conj().T,H[l],M[l])
    return Ens,M 


'''MPO for TFIM
[[1,sz,h*sx,],
 [0,0 , -sz ],
 [0,0 , 1 ]]
 On repeated multiplication, the [2][0] entry gives us the Hamiltonian. 
 '''
def tfiMPO(N,h):
    Hl = np.zeros((3,2,3,2))
    Hl[0,:,0,:] = I2
    Hl[1,:,0,:] = sz
    Hl[2,:,0,:] = h*sx
    Hl[2,:,1,:] = -sz
    Hl[2,:,2,:] = -I2
    H = [Hl for l in range(N)]
    H[0] = Hl[-1:np.shape(Hl)[0],:,:,:]
    H[N-1] = Hl[:,:,0:1,:]
    return H

#Model Parameters
N=40
D=10
h=0 
Nsweeps=3
H = tfiMPO(N,h)
En,psi = dmrg_1site_obc(H,D,Nsweeps)
print('ground state energy =', En[-1])

#MPO for local sigma_z operator:
sz_MPO = np.zeros((1,2,1,2))
sz_MPO[0,:,0,:] = sz
I2_MPO = np.zeros((1,2,1,2))
I2_MPO[0,:,0,:] = I2
ran=np.linspace(0,2,100)

corr=[]
for h in ran:
    corrs=[]
    H = tfiMPO(N,h)
    En,psi = dmrg_1site_obc(H,D,Nsweeps)
    for it in range(N-1):
        Taux = np.ones((1,1,1))
        for l in range(N):
            if l==it:
                Taux = ZipLeft(Taux,psi[l].conj().T,sz_MPO,psi[l])   
            elif l==it+1:
                Taux = ZipLeft(Taux,psi[l].conj().T,sz_MPO,psi[l])      #nearest-neighbour spin-correlation
            else:
                Taux = ZipLeft(Taux,psi[l].conj().T,I2_MPO,psi[l])
        corrs.append(Taux[0,0,0])
    corr.append(np.sum(corrs))   
plt.plot(ran,corr)  
plt.xlabel("h")
plt.ylabel("$\sigma^z_l \sigma^z_{l+1}$")
plt.show()      

ran=np.linspace(0,2,100)
mag=[]
for h in ran:
    mags=[]
    H = tfiMPO(N,h)
    En,psi = dmrg_1site_obc(H,D,Nsweeps)
    for it in range(N-1):
        Taux = np.ones((1,1,1))
        for l in range(N):
            if l==it:
                Taux = ZipLeft(Taux,psi[l].conj().T,sz_MPO,psi[l])   #local magnetisation
            else:
                Taux = ZipLeft(Taux,psi[l].conj().T,I2_MPO,psi[l])
        mags.append(Taux[0,0,0])
    mag.append(np.sum(mags)/N)   #average magnetisation

plt.plot(ran,mag)  
plt.xlabel("h")
plt.ylabel("Magnetization")
plt.show()

#Average Magnetization falls to zero in the paramagnetic phase (h>1)
#Nearest-neighbour spin correlation starts out at 1 for h=0, and falls as h is increased.
