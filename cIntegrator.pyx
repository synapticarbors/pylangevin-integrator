import numpy as np
cimport numpy as np
cimport cython
cimport ForceFields

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

include "Python.pxi"

cdef extern from "math.h":
    double sqrt(double)
 
cdef extern from "randomkit.h": 
    ctypedef struct rk_state: 
        unsigned long key[624] 
        int pos 
        int has_gauss 
        double gauss 
    void rk_seed(unsigned long seed, rk_state *state) 
    double rk_gauss(rk_state *state)


cdef class Integrator:
    cdef double _beta, _mass, _nu, _dt, _invmass, _sigma
    cdef double _b1, _b2, _s3, _sdt3, _sdt
    cdef unsigned int _dims
    cdef ForceFields.Force _ff
    cdef DTYPE_t* _Fn
    cdef rk_state *rng_state
    
    def __init__(self,ForceFields.Force forcefield, double mass, double nu, double beta, double dt, unsigned int dims, unsigned long seed):
        self._Fn = <DTYPE_t*>PyMem_Malloc(dims*sizeof(DTYPE_t))
        self.rng_state = <rk_state*>PyMem_Malloc(sizeof(rk_state)) 
        
    def __dealloc__(self): 
        if self._Fn != NULL: 
            PyMem_Free(self._Fn) 
            self._Fn = NULL
        
        if self.rng_state != NULL: 
            PyMem_Free(self.rng_state) 
            self.rng_state = NULL
    
    def __init__(self,ForceFields.Force forcefield, double mass, double nu, double beta, double dt, unsigned int dims, unsigned long seed):
        self._beta = beta
        self._mass = mass
        self._nu = nu
        self._dt = dt
        self._dims = dims

        self._ff = forcefield

        self._invmass = 1.0/mass

        self._sigma = sqrt(2.0*self._nu/(self._beta*self._mass))

        self._b1 = 1.0 - 0.5*self._dt*self._nu + 0.125*(self._dt**2)*(self._nu**2)
        self._b2 = 0.5*self._dt - 0.125*self._nu*self._dt**2

        self._s3 = sqrt(3.0)
        self._sdt3 = self._sigma*sqrt(self._dt**3)
        self._sdt = self._sigma*sqrt(self._dt)
        
        rk_seed(seed, self.rng_state)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def step(self, np.ndarray[DTYPE_t, ndim=1] coords, np.ndarray[DTYPE_t, ndim=1] velocs, int steps):
       cdef unsigned int k, d
       cdef double xi, eta, n1, n2, n3, n4
       cdef np.ndarray[DTYPE_t, ndim=1] F = np.zeros((self._dims,),dtype=DTYPE)
       cdef np.ndarray[DTYPE_t, ndim=1] n5 = np.zeros((self._dims,),dtype=DTYPE)
       
       for k in xrange(steps):
           
           for d in xrange(self._dims):
               xi = rk_gauss(self.rng_state)
               eta = rk_gauss(self.rng_state)

               n1 = 0.5*self._sdt*xi
               n3 = 0.5*self._sdt3*eta/self._s3
               n4 = self._sdt3*(0.125*xi + 0.25*eta/self._s3)
               n5[d] = n1 - self._nu*n4 
           
               velocs[d] = velocs[d]*self._b1 + self._Fn[d]*self._b2 + n5[d]
               coords[d] = coords[d] + self._dt*velocs[d] + n3
            
           self._ff.evaluate(coords,F)
            
           for d in xrange(self._dims):
               F[d] *= self._invmass
               velocs[d] = velocs[d]*self._b1 + F[d]*self._b2 + n5[d]
               self._Fn[d] = F[d]
               
	