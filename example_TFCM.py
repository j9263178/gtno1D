import numpy as np
from ADmodels.GMPOmodel import GMPOmodel
from torch import optim, tensor, eye, ones, zeros, rand, randn, matmul, kron, float64, complex64, cos, sin, real, no_grad, from_numpy, einsum
from scipy.integrate import quad

Id = eye(2, dtype=float64)
Sx = zeros(2, 2, dtype=float64); Sx[0, 1] = Sx[1, 0] = 1
Sy = zeros(2, 2, dtype=complex64); Sy[0, 1] = -1j; Sy[1, 0] = 1j
iSy = zeros(2, 2, dtype=float64); iSy[0, 1] = 1; iSy[1, 0] = -1
Sz = zeros(2, 2, dtype=float64); Sz[0, 0] = 1; Sz[1, 1] = -1
Sp = real(Sx + 1j*Sy); Sm = real(Sx - 1j*Sy)

##  Helper functions
def symmetrize(A, B):
    return (matmul(A, B) + matmul(B, A))/2

def symmetrize2(A, B, C):
    return (matmul(A, symmetrize(B,C)) + matmul(B, symmetrize(A,C)) + matmul(C, symmetrize(A,B)))/3

if __name__ == "__main__":

    ##  Construct the initial state |+...+> for TFIM.    
    def tfcm_AA(cs):
        d = 2
        A = zeros(d, dtype=float64)
        A[0] = A[1] = 1.
        AA = kron(A, A)
        return AA

    ##  Construct the local TFCM Hamiltonian for given parameter g. 
    def tfcm_h(g):
        zxz = 0.5*(einsum("ae,bf,cg,dh->abcdefgh",Sz,Sx,Sz,Id)+einsum("ae,bf,cg,dh->abcdefgh",Id,Sz,Sx,Sz))
        x = (0.25)*einsum("ae,bf,cg,dh->abcdefgh",Sx,Id,Id,Id)
        x += (0.25)*einsum("ae,bf,cg,dh->abcdefgh",Id,Sx,Id,Id)
        x += (0.25)*einsum("ae,bf,cg,dh->abcdefgh",Id,Id,Sx,Id)
        x += (0.25)*einsum("ae,bf,cg,dh->abcdefgh",Id,Id,Id,Sx)
        zxz = zxz.reshape(4,4,4,4)
        x = x.reshape(4,4,4,4)    
        h = (-g*x-zxz)
        return h
        
    ##  Construct gmpo for TFCM.
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
        for a in range(len(idxs)):
            for b in range(a,len(idxs)):
                if a == 0 and b == 0:
                    # G[:, :, 0, 0] += cs[m]*(onsites[0] + onsites[1]) + cs[m+1]*onsites[2]
                    G[:, :, 0, 0] += cs[m]*(onsites[0] + onsites[1] + onsites[2])
                    m += 1
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
        return G
    
    ##  Closure function for optimizer 
    def closure():
        # E0 = model.evaluate_E()
        E0 = model.evaluate_E_sparse() # for large D one should use sparse forward.
        optimizer.zero_grad()
        E0.backward()
        return E0
    
    ##  Get the exact TFCM ground state energy for a given g   
    def get_E0_exact(hx):
        def integrand(k):
            epsilon = np.cos(2*k) - hx; 
            delta = np.sin(2*k)
            return np.sqrt(epsilon**2 + delta**2)
        E0 = -quad(lambda k: integrand(k), 0, np.pi)[0]/np.pi
        return E0
    
    numG = 1; numc_G = 11; numA = 0
    lenc = numG*numc_G + numA

    epochnum = 10
    cinit = rand(lenc,dtype=float64)
    
    ## for numG = 1 we start with h_x = 0, for higher numG we only scale around the critical point   
    if numG == 1:
        gs = np.concatenate([np.linspace(0, 0.76, 31, endpoint=False), np.concatenate([np.linspace(0.76, 0.93, 30, endpoint=False), np.linspace(0.93, 2, 31)])])
        cinit = tensor(5*[1]+6*[0],dtype=float64)
    if numG == 2:
        gs = np.concatenate([np.linspace(0.8, 0.93, 5, endpoint=False), np.concatenate([np.linspace(0.93, 1.00, 30, endpoint=False), np.linspace(1.00, 1.05, 5)])])

    ##  Initialize the GMPOmodel   
    model = GMPOmodel(localh = tfcm_h, gmpo = tfcm_G, A = tfcm_AA, numG = numG, Aparas = numA, Gparas = numc_G) 
    model.setcs(cs = cinit)
    model.setreqgrad(reqgrad = lenc*[1])
    
    ##  Initialize the optimizer  
    optimizer = optim.LBFGS(model.parameters(), max_iter=40, tolerance_grad = 0, tolerance_change = 0, line_search_fn="strong_wolfe")

    ##  File reading and writing labels
    obsfn = f"datas/tfcm_numG{numG}_obs.txt"
    csfn = f"datas/tfcm_numG{numG}_cs.txt"
        
    obsf = open(obsfn, "a")
    obsf.write("# hx E0 deltaE VOP\n")
    obsf.close()
    
    csf = open(csfn, "a")
    first = "# hx"
    for i in range(1, lenc):
        first += (" c" + str(i) )
    csf.write(first+"\n")
    csf.close()

    ## optimize GTNO and measure observables for each h_x
    for g in gs:
        print(f"h_x = {g}")
        model.g = g
        E0_exact = get_E0_exact(model.g)
        for epoch in range(epochnum):
            E0 = optimizer.step(closure)
            print(f"deltaE = {(E0.item()-E0_exact)/abs(E0_exact)}")

        with no_grad():
            
            deltaE = (E0.item()-E0_exact)/abs(E0_exact)
            
            ## evaluate VOP
            A = model.get_gmpoA()
            u = kron(Sx,Id); V = kron(Sx,Sz)
            for _ in range(numG-1):
                V = kron(V,kron(Id,Sz))
            uA = einsum("ai,ibc->abc",u, A)
            VuAV = einsum("bi,aij,jc->abc", V.T, uA, V) 
            print(f"norm(A-VuAV)/A.norm() = {(A-VuAV).norm()/A.norm()}")
            u = kron(Id,Sx); V = kron(Id,Sx)
            for _ in range(numG-1):
                V = kron(V,kron(Sz,Id))
            uA = einsum("ai,ibc->abc",u, A)
            VuAV = einsum("bi,aij,jc->abc", V.T, uA, V)  
            print(f"norm(A-VuAV)/A.norm() = {(A-VuAV).norm()/A.norm()}")
            VOP = (A-VuAV).norm()/A.norm()
            
            ## write to observable file
            tmp = np.asarray([model.g, E0.item(), deltaE, VOP])
            with open(obsfn, "a") as f:
                np.savetxt(f, tmp.reshape(1, tmp.shape[0]))
            f.close() 
            print(f"h_x = {tmp[0]}, deltaE = {tmp[2]}, VOP = {tmp[3]}")
            
            ## write to cs file                         
            tmp = np.asarray(model.getcsarray()); tmp = np.insert(tmp, 0, model.g)
            with open(csfn, "a") as f:
                np.savetxt(f, tmp.reshape(1, tmp.shape[0]))
            f.close()


