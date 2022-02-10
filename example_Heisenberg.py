"""
    Construction of GTNO, initial states A and local Hamitonian for studying the 1D Heisenberg model,
    followed by the energy optimization, here we try 20 random initial guesses and record the best energy.
"""

import numpy as np
from ADmodels.GMPOmodel import GMPOmodel
from torch import optim, einsum, tensor, eye, ones, zeros, rand, randn, matmul, kron, float64, complex64, cos, sin, real

Id = eye(2, dtype=float64)
Sx = zeros(2, 2, dtype=float64); Sx[0, 1] = Sx[1, 0] = 1
Sy = zeros(2, 2, dtype=complex64); Sy[0, 1] = -1j; Sy[1, 0] = 1j
iSy = zeros(2, 2, dtype=float64); iSy[0, 1] = 1; iSy[1, 0] = -1
Sz = zeros(2, 2, dtype=float64); Sz[0, 0] = 1; Sz[1, 1] = -1
Sp = real(Sx + 1j*Sy); Sm = real(Sx - 1j*Sy)

def symmetrize(A, B):
    return (matmul(A, B) + matmul(B, A))/2

def symmetrize2(A, B, C):
    return (matmul(A, symmetrize(B,C)) + matmul(B, symmetrize(A,C)) + matmul(C, symmetrize(A,B)))/3

if __name__ == "__main__":
    
    def singlet_AA(cs):
        D = 2; d = 4
        o, x = zeros(2, dtype=float64), zeros(2, dtype=float64)
        o[0] = 1; x[1] = 1
        oo = kron(o, o); ox = kron(o, x); xo = kron(x, o); xx = kron(x, x)
        AA = zeros(d, dtype=float64)
        AA = ox-xo
        return AA
    
    def majumdar_ghosh_state(cs):
        o, x = zeros(2, dtype=float64), zeros(2, dtype=float64)
        o[0] = 1; x[1] = 1
        oo = kron(o, o); ox = kron(o, x); xo = kron(x, o); xx = kron(x, x)
        MG = zeros(2,3,3, dtype=float64);
        MG[:,0,2] = o; MG[:,2,1] = -o;
        MG[:,1,2] = MG[:,2,0] = x;
        MG = einsum("aci,bid->abcd", MG, MG).reshape(4,3,3)
        return MG
        
    def heisenberg_G(cs):

        d, D = 2, 4
        G = zeros(d,d,D,D,dtype=float64)
        rowops = [Id,Sm/2**0.5,Sp/2**0.5,Sz]
        columnops = [Id,Sp/2**0.5,Sm/2**0.5,Sz]
        idxlist = [[0],[1,2,3]]

        c1 = abs(cos(cs[0]*np.pi)); c2 = sin(cs[0]*np.pi)*cos(cs[1]*np.pi); c3 = sin(cs[0]*np.pi)*sin(cs[1]*np.pi)

        for a in idxlist:
            for b in idxlist:
                if a == [0] and b == [0]:
                    G[:, :, 0, 0] = c1*Id
                else:
                    for i in a:
                        for j in b:
                            optmp = symmetrize(rowops[i], columnops[j])
                            if i == j:
                                G[:, :, i, j] = c3*optmp
                            elif i > j: 
                                G[:, :, i, j] = c2*optmp
                            else: G[:, :, i, j] = abs(c2)*optmp

        return G

    def heisenberg_G_twosites(cs):
        G = heisenberg_G(cs)
        GG = einsum("acei,bdif->abcdef", G, G).reshape(4,4,4,4)
        return GG

    def heisenberg_h_twosites(g):
        d = 4
        h_temp = 0.25*real(kron(Sx, Sx) + kron(Sy, Sy) + kron(Sz, Sz))
        h = 0.25*(kron(h_temp, eye(4)) +  kron(eye(4), h_temp)
                 +2*kron(eye(2), kron(h_temp, eye(2)))).reshape(d,d,d,d)
        return h
    
    def closure():
        # E0 = model.evaluate_E()
        E0 = model.evaluate_E_sparse() # for large D one should use sparse forward.
        optimizer.zero_grad()
        E0.backward()
        return E0
    
    E0_exact = (1/4 - np.log(2))
    
    numG = 2; numc_G = 2
    lenc = numG*numc_G
    model = GMPOmodel(localh = heisenberg_h_twosites, gmpo = heisenberg_G_twosites, A = singlet_AA,
                        numG = numG, Aparas = 0, Gparas = 2)
    
    epochnum = 5
    bestE = 999
        
    for trial in range(20):
        model.setcs(cs = rand(lenc, dtype=float64))
        model.setreqgrad(reqgrad = ones(lenc))

        optimizer = optim.LBFGS(model.parameters(), max_iter=40, tolerance_grad = 0, tolerance_change = 0, line_search_fn="strong_wolfe")

        for epoch in range(epochnum):
            E0 = optimizer.step(closure)
            
        print("Trial = ", trial," dE = ", (E0.item()-E0_exact), f"dE/E = {(E0.item()-E0_exact)/abs(E0_exact)}")
        
        if E0.item()<bestE:
            bestE = E0.item() 
            bestcs = model.getcsarray()
            
    print("-"*30)
    print("Best E = ", bestE," dE = ", (bestE-E0_exact), f"dE/E = {(bestE-E0_exact)/abs(E0_exact)}")
    print(bestcs)
    print("-"*30)
