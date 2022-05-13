import numpy as np
from ADmodels.GMPOmodel import GMPOmodel
from torch import optim, tensor, eye, ones, zeros, rand, randn, matmul, kron, float64, complex64, cos, sin, real, no_grad, from_numpy, einsum
import scipy.integrate as integrate

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
    def ising_A(cs):
        A = zeros(2, dtype=float64)
        A[0] = 1; A[1] = 1
        return A
    
    ##  Construct the local TFIM Hamiltonian for given parameter g. 
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

    ##  Construct gmpo for TFIM.
    def tfim_G(cs):
        D, d = 2, 2
        G = zeros(d, d, D, D, dtype=float64)
        G[:, :, 0, 0] = Id+cs[2]*Sx
        G[:, :, 0, 1] = G[:, :, 1, 0] = cs[0]*Sz
        G[:, :, 1, 1] = cs[1]*Id+cs[3]*Sx/3
        return G
    
    ##  Closure function for optimizer
    def closure():
        # E0 = model.evaluate_E()
        E0 = model.evaluate_E_sparse() # for large D one should use sparse forward.
        optimizer.zero_grad()
        E0.backward()
        return E0

    ##  Get the exact TFIM ground state energy for a given g   
    def get_E0_exact(g):
        f = lambda k: -1. / (2. * np.pi) * np.sqrt(g**2 - 2 * g * np.cos(k) + 1)
        E0, error = integrate.quadrature(f, -np.pi, np.pi, tol=1e-15, rtol=1e-15, maxiter=1800)
        return E0

    epochnum = 10

    numG = 1; numc_G = 4; numA = 0
    lenc = numG*numc_G+numA
    
    cinit = rand(lenc, dtype = float64)

    ## for numG = 1 we start with h_x = 0, for higher numG we only scale around the critical point
    if numG == 1:
        cinit = tensor([1, 1, 0, 0], dtype=float64)
        gs = np.concatenate([np.linspace(0.0, 0.70, 13), np.concatenate([np.linspace(0.70, 1.00, 31), np.linspace(1.00, 2.00, 13)])])

    if numG == 2:
        gs = np.concatenate([np.linspace(0.75, 0.93, 8), np.concatenate([np.linspace(0.93, 1.00, 31), np.linspace(1.00, 1.20, 13)])])
        
    if numG == 3:
        gs = np.concatenate([np.linspace(0.75, 0.97, 13), np.concatenate([np.linspace(0.97, 1.01, 31), np.linspace(1.01, 1.20, 13)])])

    if numG == 4:
        gs = np.concatenate([np.linspace(0.75, 0.98, 13), np.concatenate([np.linspace(0.98, 1.025, 31), np.linspace(1.025, 1.20, 13)])])
    

    ##  Initialize the GMPOmodel  
    model = GMPOmodel(localh = tfim_h, gmpo = tfim_G, A = ising_A, numG = numG, Aparas = numA, Gparas = numc_G)        
    model.setcs(cs = cinit)
    model.setreqgrad(reqgrad = lenc*[1]) #G
 
 
    ##  Initialize the optimizer       
    optimizer = optim.LBFGS(model.parameters(), max_iter = 20, tolerance_grad = 0, tolerance_change = 0, line_search_fn="strong_wolfe")


    ##  File reading and writing labels
    obsfn = f"datas/tfim_numG{numG}_obs.txt"
    csfn = f"datas/tfim_numG{numG}_cs.txt"
        
    obsf = open(obsfn, "a")
    obsf.write("# hx E0 deltaE VOP EOP\n")
    obsf.close()
    
    csf = open(csfn, "a")
    first = "# hx "
    for i in range(1, lenc):
        first += (" c" + str(i) )
    csf.write(first+"\n")
    csf.close()

    ## construct the observable for VOP    
    u = Id; V = Sx
    for _ in range(numG-1):
        V = kron(V,Id)
        
    ## construct the observable for EOP
    Sz_ = Sz
    Id_ = Id
    for _ in range(numG-1):
        Sz_ = kron(Sz_,  Id)
        Id_ = kron(Id_, Id)
    ZI = kron(Sz_, Id_)
    
    ## optimize GTNO and measure observables for each h_x
    for g in gs:
        print(f"h_x = {g}")
        model.g = g
        exactE0 = get_E0_exact(model.g)
        for epoch in range(epochnum):
            E0 = optimizer.step(closure)
            print(f"deltaE = {(E0.item()-exactE0)/abs(exactE0)}")
            
        with no_grad():
            
            deltaE = (E0.item()-exactE0)/abs(exactE0)   
                  
            A = model.get_gmpoA().detach()    
                    
            ## evaluate EOP
            EOP = model.get_EOP(ZI)
        
            ## evaluate VOP
            uA = einsum("ai,jbc->abc",u, A)
            VuAV = einsum("bi,aij,jc->abc", V.T, uA, V)
            VOP = (A-VuAV).norm()/A.norm() 
            
            ## write to observable file
            tmp = np.asarray([model.g, E0.item(), deltaE, VOP, EOP])
            with open(obsfn, "a") as f:
                np.savetxt(f, tmp.reshape(1, tmp.shape[0]))
            f.close() 
            print(f"h_x = {tmp[0]}, deltaE = {tmp[2]}, VOP = {tmp[3]}, EOP = {tmp[4]}")
            
            ## write to cs file            
            tmp = np.asarray(model.getcsarray()); tmp = np.insert(tmp, 0, model.g)
            with open(csfn, "a") as f:
                np.savetxt(f, tmp.reshape(1, tmp.shape[0]))
            f.close()