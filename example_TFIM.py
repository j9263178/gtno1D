"""
    Construction of GTNO, initial states A and local Hamitonian for studying the 1D traversed field ising model,
    followed by the energy optimization at different traversed field values g. We do it for the numG = 2 and 3 here.
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
    
    def ising_A(cs):
        d, D = 2, 2
        A = zeros(d, dtype=float64)
        A[0] = cos(cs[0]); A[1] = sin(cs[0])  
        return A
    
    def tfim_h(g):   
        d = 2
        h = zeros(d, d, d, d, dtype=float64)
        h[0, 0, 0, 0] = h[1, 1, 1, 1] = -1.0
        h[0, 1, 0, 1] = h[1, 0, 1, 0] = 1.0
        h[1, 0, 0, 0] = h[0, 1, 0, 0] = \
        h[1, 1, 0, 1] = h[0, 0, 0, 1] = \
        h[0, 0, 1, 0] = h[1, 1, 1, 0] = \
        h[0, 1, 1, 1] = h[1, 0, 1, 1] = -g/2
        return h

    def tfim_G(cs):
        D, d = 2, 2
        G = zeros(d, d, D, D, dtype=float64)
        G[:, :, 0, 0] = Id+cs[0]*Sx
        G[:, :, 0, 1] = G[:, :, 1, 0] = cs[1]*Sz
        G[:, :, 1, 1] = cs[2]*Id+cs[3]*Sx
        return G

    def closure():
        # E0 = model.evaluate_E()
        E0 = model.evaluate_E_sparse() # for large D one should use sparse forward.
        optimizer.zero_grad()
        E0.backward()
        return E0
       
    epochnum = 20
    
    data = np.load("datas/tfim_E0_exact.npz")
    exactE0 = data["E0s"]
    gs = data["gs"]

    for numG in [2, 3]:
        model = GMPOmodel(localh = tfim_h, gmpo = tfim_G, A = ising_A, numG = numG, Aparas = 1, Gparas = 4)
        model.setcs(cs = 0.9*ones(1+4*numG, dtype=float64))
        model.setreqgrad(reqgrad = ones(1+4*numG))
        optimizer = optim.LBFGS(model.parameters(), max_iter = 40, tolerance_grad = 0, tolerance_change = 0, line_search_fn="strong_wolfe")
    
        E0s = []
        c0s = []
        for i in range(gs.size):
            model.g = gs[i]
            for epoch in range(epochnum):
                E0 = optimizer.step(closure)
                
            print("g = ", gs[i]," dE = ", (E0.item()-exactE0[i]))
            
            E0s.append(E0.item())
            c0s.append(model.getcsarray()[0])

        np.savez(f"datas/tfim_E0s_{numG}.npz", gs = gs, E0s = np.asarray(E0s))
        np.savez(f"datas/tfim_c0s_{numG}.npz", gs = gs, c0s = np.asarray(c0s))
