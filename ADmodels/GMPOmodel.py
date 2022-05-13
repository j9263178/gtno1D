import torch
import numpy as np
import scipy.sparse.linalg as sparselinalg
from DominantSparseEigenAD.eig import DominantEig
import DominantSparseEigenAD.eig as DSEADeig


class GMPOmodel(torch.nn.Module):
    
    """
        localh: Function that returns a two-sites hamitonian (d, d, d, d)
        gmpo: Function that returns a GMPO (d, d, D, D) 
        A: Function that returns a state that GMPO will apply to (d) or (d, D, D)
        numG: # of times we apply GMPO   
        
        Aparas: # of paras in A
        Gparas: # of paras in gmpo
    """
    
    def __init__(self, localh, gmpo, A, numG, Aparas, Gparas):

        super(GMPOmodel, self).__init__()
        self.dominant_eig = DominantEig.apply
        self.numG = numG
        self.h = localh
        self.gmpo = gmpo
        self.A = A 
        self.g = 0
        self.cs = torch.nn.ParameterList()
        self.Aparas = Aparas
        self.Gparas = Gparas
        self.ncv = None
   
    def setcs(self, cs = None):
        
        """
            Set the trainable parameters.
        """
        
        if len(self.cs) == 0:
            for i in range(len(cs)):
                self.cs.append(torch.nn.Parameter(cs[i].clone()))
        else:
            i = 0
            for para in self.cs.parameters():
                para.data = cs[i].clone()
                i+=1
            
    def setreqgrad(self, reqgrad):
        for i in range(len(self.cs)):
            if reqgrad[i] == 0:
                req = False
            else:
                req = True
            self.cs[i].requires_grad = req
     
    def getcsarray(self):
        csclone = torch.zeros(len(self.cs))
        i = 0
        for para in self.cs.parameters():
            csclone[i] = para.data.clone()
            i+=1
        return csclone
     
    def get_gmpoA(self):
        
        """
            Apply gmpo "numG" times to the initial state A.
        """
        
        gmpoA = self.A(self.cs)
        if len(gmpoA.shape) == 1:
            curD = 1
        else:
            curD = gmpoA.shape[2]
        ind = self.Aparas
        for i in range(self.numG):
            G = self.gmpo(self.cs[ind:ind+self.Gparas])
            d = G.shape[0]
            curD *= G.shape[2]
            if i==0 and len(gmpoA.shape)==1:
                gmpoA = torch.einsum('aibc,i->abc',G, gmpoA).reshape(d, curD, curD)
            else:
                gmpoA = torch.einsum('aice,ibd->abcde',G, gmpoA).reshape(d, curD, curD)
            ind += self.Gparas
        return gmpoA

    def evaluate_E(self):   
        """
            Forward pass by explicitly constructing the transfer matrix "Tsf" and evaluate energy.
        """
        gmpoA = self.get_gmpoA()
        D = gmpoA.shape[1]
        Tsf = torch.einsum("kij,kmn->imjn", gmpoA, gmpoA).reshape(D**2, D**2)
        eigval_max, leigvector_max, reigvector_max = self.dominant_eig(Tsf, self.ncv)
        
        assert 1e-4<torch.norm(leigvector_max)<1e10
        assert 1e-4<torch.norm(reigvector_max)<1e10
        
        leigvector_max = leigvector_max.reshape(D, D)
        reigvector_max = reigvector_max.reshape(D, D)
        E0 = torch.einsum("aik,bkj,abcd,cml,dln,im,jn", gmpoA, gmpoA, self.h(self.g), 
                gmpoA, gmpoA, leigvector_max, reigvector_max) / eigval_max**2
        return E0

    def _setsparsefunctions(self):
        A = self.A_.detach().numpy()
        self.D = A.shape[-1]
        self.d = A.shape[0]
        def fr(v):
            r = v.reshape(self.D, self.D)
            return np.einsum("kij,kmn,jn->im", A, A, r, optimize="greedy")
        self.Tsf = sparselinalg.LinearOperator((self.D**2, self.D**2), matvec=fr)
        def fl(v):
            l = v.reshape(self.D, self.D)
            return np.einsum("kij,kmn,im->jn", A, A, l, optimize="greedy")
        self.TsfT = sparselinalg.LinearOperator((self.D**2, self.D**2), matvec=fl)
        def Tsfadjoint_to_Aadjoint(grad_Tsf):
            grad_A = np.zeros((self.d, self.D, self.D))
            for u, v in grad_Tsf:
                umat, vmat = u.reshape(self.D, self.D), v.reshape(self.D, self.D)
                grad_A = grad_A \
                    + np.einsum("im,jn,kmn->kij", umat, vmat, A, optimize="greedy") \
                    + np.einsum("mi,nj,kmn->kij", umat, vmat, A, optimize="greedy")
            return torch.from_numpy(grad_A)
        self.Tsfadjoint_to_Aadjoint = Tsfadjoint_to_Aadjoint

    def _h_optcontraction(self, l, r, h):
        upperleft = torch.einsum("aik,im->amk", self.A_, l)
        upperright = torch.einsum("bkj,jn->bkn", self.A_, r)
        upper = torch.einsum("amk,bkn->abmn", upperleft, upperright)
        lower = torch.einsum("cml,dln->cdmn", self.A_, self.A_)
        upperlower = torch.einsum("abmn,cdmn->abcd", upper, lower)
        result = torch.einsum("abcd,abcd", upperlower, self.h(self.g))
        return result

    def evaluate_E_sparse(self):
        """
            Forward pass by treating the transfer matrix "Tsf" as a "sparse matrix"
        represented by scipy.sparse.linalg.LinearOperator. This way, various tensor
        contractions involved in the forward and backward pass can be significantly
        optimized and accelerated.
        """

        self.A_ = self.get_gmpoA()
        self._setsparsefunctions()
        
        DSEADeig.setDominantSparseEig(self.Tsf, self.TsfT, self.Tsfadjoint_to_Aadjoint)
        dominant_sparse_eig = DSEADeig.DominantSparseEig.apply 
        eigval_max, leigvector_max, reigvector_max = dominant_sparse_eig(self.A_, self.ncv)

        assert 1e-4<torch.norm(leigvector_max)<1e10
        assert 1e-4<torch.norm(reigvector_max)<1e10
        
        leigvector_max = leigvector_max.reshape(self.D, self.D)
        reigvector_max = reigvector_max.reshape(self.D, self.D)
        E0 = self._h_optcontraction(leigvector_max, reigvector_max, self.h) / eigval_max**2
    
        return E0
    
    @torch.no_grad()
    def evaluate_exp(self, O):
        gmpoA = self.get_gmpoA()
        D = gmpoA.shape[1]
        Tsf = torch.einsum("kij,kmn->imjn", gmpoA, gmpoA).reshape(D**2, D**2)
        eigval_max, leigvector_max, reigvector_max = self.dominant_eig(Tsf, self.ncv)
        assert 1e-4<torch.norm(leigvector_max)<1e10
        assert 1e-4<torch.norm(reigvector_max)<1e10
        leigvector_max = leigvector_max.reshape(D, D)
        reigvector_max = reigvector_max.reshape(D, D)
        E0 = torch.einsum("aik,bkj,abcd,cml,dln,im,jn", gmpoA, gmpoA, O,
                gmpoA, gmpoA, leigvector_max, reigvector_max) / eigval_max**2
        return E0

    @torch.no_grad()
    def get_EOP(self, obs):
        gmpoA = self.get_gmpoA().detach()
        D = gmpoA.shape[1]
        Tsf = torch.einsum("kij,kmn->imjn", gmpoA, gmpoA).reshape(D**2, D**2)
        eigval_max, leigvector_max, reigvector_max = self.dominant_eig(Tsf, self.ncv)
        assert 1e-4<torch.norm(leigvector_max)<1e10
        assert 1e-4<torch.norm(reigvector_max)<1e10
        res = torch.einsum("ij,i,j", obs, leigvector_max, reigvector_max)
        return res