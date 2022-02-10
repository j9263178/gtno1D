"""
    Construction of GTNO, initial states A and local Hamitonian for studying the 1D traversed field cluster model,
    followed by the energy optimization at different traversed field values g. We do it for numG = 1 and evaluate
    the virtual order parameter and other observables.
"""

import numpy as np
from scipy.integrate import quad
from ADmodels.GMPOmodel import GMPOmodel
from torch import optim, einsum, tensor, eye, ones, zeros, rand, randn, matmul, kron, float64, complex64, cos, sin, real, no_grad

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

zxz = (einsum('ae,bf,cg,dh->abcdefgh',Sz,Sx,Sz,Id)+einsum('ae,bf,cg,dh->abcdefgh',Id,Sz,Sx,Sz))/2
x = (0.25)*einsum('ae,bf,cg,dh->abcdefgh',Sx,Id,Id,Id)
x += (0.25)*einsum('ae,bf,cg,dh->abcdefgh',Id,Sx,Id,Id)
x += (0.25)*einsum('ae,bf,cg,dh->abcdefgh',Id,Id,Sx,Id)
x += (0.25)*einsum('ae,bf,cg,dh->abcdefgh',Id,Id,Id,Sx)
zxz = zxz.reshape(4,4,4,4)
x = x.reshape(4,4,4,4)

if __name__ == "__main__":
    
    def tfcm_AA(cs):
        d = 2
        A = zeros(d, dtype=float64)
        A[0] = A[1] = 1.
        AA = kron(A, A)
        return AA
    
    def tfcm_G(cs):  
        D = 4; d = 4
        G = zeros(d, d, D, D, dtype=float64)
        rowops = [kron(Id, Id), kron(Sz, Id), kron(Sx, Sz), kron(iSy, Sz)]
        columnops = [kron(Id, Id), kron(Sz, Sx), kron(Id, Sz), kron(-Sz, iSy)]       
        idxs = [[0],[1,2],[3]]
        onsites = [kron(Id, Sx), kron(Sx, Id), kron(Sx, Sx)]
        m = 0
        for a in range(len(idxs)):
            for b in range(a,len(idxs)):
                # print(f"a = {a}, b = {b}")
                if a == 0 and b == 0:
                    G[:, :, 0, 0] = kron(Id, Id) 
                else:
                    for i in idxs[a]:
                        for j in idxs[b]:
                            optmp = symmetrize(rowops[i],columnops[j])
                            G[:, :, i, j] =  abs(cs[m])*optmp
                            optmp = symmetrize(rowops[j],columnops[i])
                            G[:, :, j, i] = (cs[m])*optmp
                    m += 1
        # print(f"num of cs = {m}")
        # exit()
        for a in range(len(idxs)):
            for b in range(a,len(idxs)):
                if a == 0 and b == 0:
                    G[:, :, 0, 0] += cs[m]*(onsites[0] + onsites[1]) + cs[m+1]*onsites[2]
                    m +=2
                else:               
                    for i in idxs[a]:
                        for j in idxs[b]:
                            for k in range(3):
                                onsitetmp = symmetrize2(onsites[k],rowops[i],columnops[j])
                                G[:, :, i, j] += cs[m]*onsitetmp
                                onsitetmp = symmetrize2(onsites[k],rowops[j],columnops[i]) 
                                G[:, :, j, i] += cs[m]*onsitetmp
                    m += 1
        # print(f"num of cs = {m}")
        # exit()
        return G

    def tfcm_h(g):
        h = (-g*x-zxz)
        return h
    
    def closure():
        # E0 = model.evaluate_E()
        E0 = model.evaluate_E_sparse() # for large D one should use sparse forward.
        optimizer.zero_grad()
        E0.backward()
        return E0

    def get_En_exact(hx):
        def integrand(k):
            epsilon = np.cos(2*k) - hx; 
            delta = np.sin(2*k)
            return np.sqrt(epsilon**2 + delta**2)
        En = -quad(lambda k: integrand(k), 0, np.pi)[0]/np.pi
        return En
    
    numG = 1; numc_G = 12; numA = 0
    lencG = numG*numc_G
    model = GMPOmodel(localh = tfcm_h, gmpo = tfcm_G, A = tfcm_AA, numG = numG, Aparas = numA, Gparas = numc_G)
    
    epochnum = 10
    gs1 = np.linspace(0, 0.76, 31, endpoint=False)
    gs2 = np.linspace(0.76, 0.93, 20, endpoint=False)
    gs3 = np.linspace(0.93, 2, 31)
    gs = np.concatenate([gs1, np.concatenate([gs2, gs3])])

    cinit = tensor(5*[1]+(numc_G-5)*[0],dtype=float64)
    model.setcs(cs = cinit)
    model.setreqgrad(reqgrad = ones(numA+lencG))
    
    E0s, dE_Es, cs, zxzs, xs, vops = [], [], [], [], [], []
    
    for i in range(gs.size):
        model.g = gs[i]
        optimizer = optim.LBFGS(model.parameters(), max_iter=50, tolerance_grad = 0, tolerance_change = 0, line_search_fn="strong_wolfe")

        En_exact = get_En_exact(model.g)
        for epoch in range(epochnum):
            E0 = optimizer.step(closure)
        print("-"*50)
        dE_E = (E0.item()-En_exact)/abs(En_exact)
        print("g = ", model.g," dE/E = ", dE_E)
        print(model.getcsarray())
        
        with no_grad():
            
            E0s.append(E0.item()); dE_Es.append(dE_E)
            cs.append(model.getcsarray().numpy())
            zxzs.append(model.evaluate_exp(zxz).item())
            xs.append(model.evaluate_exp(x).item())
            
            # Below evaluates VOPs
            
            A = model.get_gmpoA()
            
            u = kron(Sx,Id); V = kron(Sx,Sz)
            for _ in range(numG-1):
                V = kron(V,kron(Id,Sz))
            uA = einsum("ai,ibc->abc",u,A)
            VuAV = einsum("bj,ajk,kc->abc",V.T,uA,V)             
            print(f"norm(A-VuAV)/A.norm() = {(A-VuAV).norm()/A.norm()}")
            
            u = kron(Id,Sx); V = kron(Id,Sx)
            for _ in range(numG-1):
                V = kron(V,kron(Sz,Id))
            uA = einsum("ai,ibc->abc",u,A)
            VuAV = einsum("bj,ajk,kc->abc",V.T,uA,V)  
            print(f"norm(A-VuAV)/A.norm() = {(A-VuAV).norm()/A.norm()}")  
            
            vops.append((A-VuAV).norm()/A.norm().item())

    np.savez(f"datas/tfcm_datas_{numG}.npz", gs = gs, E0s = np.asarray(E0s),dE_Es = np.asarray(dE_Es), zxzs = np.asarray(zxzs), xs= np.asarray(xs) , vops = np.asarray(vops))
    np.savez(f"datas/tfcm_cs_{numG}.npz", gs = gs, cs = np.asarray(cs))
    


