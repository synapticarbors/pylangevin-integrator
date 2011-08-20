import numpy as np
cimport numpy as np

cdef extern from "math.h":
    double exp(double)
    double cos(double)
    double sin(double)

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef class Force:
    cpdef int evaluate(self,np.ndarray[DTYPE_t, ndim=1] coords, np.ndarray[DTYPE_t, ndim=1] force)