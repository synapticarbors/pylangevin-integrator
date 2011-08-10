import numpy as np
cimport numpy as np

cimport ForceFields
cimport cython

DTYPE = np.float64

cdef class Force:
    cpdef int evaluate(self,np.ndarray[DTYPE_t, ndim=1] coords, np.ndarray[DTYPE_t, ndim=1] force):
        return 0

cdef class RouxPanForce(Force):

    @cython.boundscheck(False)
    @cython.wraparound(False) 
    cpdef int evaluate(self,np.ndarray[DTYPE_t, ndim=1] coords, np.ndarray[DTYPE_t, ndim=1] force):
        cdef double x,y,x2,y2,t1,t2,t3,fx,fy
        
        x = coords[0]
        y = coords[1]

        x2 = x**2
        y2 = y**2
        t1 = exp(-y2 - (1 - x)**2)
        t2 = exp(-y2 - (1 + x)**2)
        t3 = exp(-(8.0/25)*(x2 + y2 + 20.0*(x + y)**2))

        fx = -3.0*(2.0 - 2.0*x)*t1 \
            + 3.0*(2 + 2*x)*t2 \
            - 15.0*(8.0/25)*(2*x + 20.0*(2*x + 2*y))*t3 \
            + 4*(32.0/625)*x**3


        fy = 6.0*y*t1 \
            + 6.0*y*t2 \
            -15.0*(8.0/25)*(2*y + 20.0*(2*x + 2*y))*t3 \
            + 4*(32.0/625)*y**3 \
            -4.0*(2.0/5)*exp(-2.0 - 4.0*y)
            
        force[0] = -fx
        force[1] = -fy
        
        return 0
        
cdef class MuellerForce(Force):

    @cython.boundscheck(False)
    @cython.wraparound(False) 
    cpdef int evaluate(self,np.ndarray[DTYPE_t, ndim=1] coords, np.ndarray[DTYPE_t, ndim=1] force):
        cdef double x,y,fx,fy,b, xt, yt
        
        # Parameters
        #ai[4] = {-1.0,  -1.0,  -6.5,  0.7}
        #bi[4] = { 0.0,  0.0,   11.0,  0.6}
        #ci[4] = {-10.0, -10.0, -6.5,  0.7}

        #xi[4] = {1.0,   0.0,   -0.5, -1.0}
        #yi[4] = {0.0,   0.5,    1.5,  1.0}

        #Ai[4] = {-200.0,-100.0,-170.0,15.0}

        x = coords[0]
        y = coords[1]
        
        fx = 0
        fy = 0
        
        # 0
        b = -200.0*exp(-(x-1.0)**2 - 10.0*y**2)
        fx += -2.0*(x-1.0)*b
        fy += -20.0*y*b
        
        # 1
        b = -100.0*exp(-x**2 -10.0*(y-0.5)**2)
        fx += -2.0*x*b
        fy += -20.0*(y-0.5)*b
        
        #2
        xt = x+0.5
        yt = y-1.5
        b = -170.0*exp(-6.5*xt**2 + 11.0*xt*yt -6.5*yt**2)
        fx += (-13.0*xt + 11.0*yt)*b
        fy += (11.0*xt -13.0*yt)*b
        
        #3
        xt = x+1
        yt = y-1
        b = 15.0*exp(0.7*xt**2 + 0.6*xt*yt + 0.7*yt**2)
        fx += (1.4*xt + 0.6*yt)*b
        fy += (0.6*xt + 1.4*yt)*b 
        
        
        force[0] = -fx
        force[1] = -fy
        
        return 0
    

