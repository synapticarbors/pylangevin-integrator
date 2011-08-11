pylangevin-integrator
======================

Author:   Joshua L. Adelman, University of Pittsburgh

2nd Order Langevin integrator written in cython/python. The code is based on Eqn 23 of:

Second-order integrators for Langevin equations with holonomic constraints  
Eric Vanden-Eijnden and Giovanni Ciccotti  
_Chemical Physics Letters_ (2006) 429, 310-316  
[http://dx.doi.org/10.1016/j.cplett.2006.07.086](http://dx.doi.org/10.1016/j.cplett.2006.07.086)  

If you use this code in a publication, please cite the above paper.

Installation
---------------

This module requires:  
numpy [http://numpy.scipy.org/](http://numpy.scipy.org/)  
cython (optional) [http://cython.org/](http://cython.org/)  
gcc [http://gcc.gnu.org/](http://gcc.gnu.org/)  

To compile the extension in the directory in which *.pyx and *.c reside:  
    
    python setup.py build_ext --inplace

If `setup.py` detects that you do not have `cython` installed, it will build the modules from the *.c files. You will, however, need
cython if you intend to modify or add force fields since you will have to re-cythonize the source.

For further information on compiling cython extensions see:  
[http://docs.cython.org/src/userguide/source_files_and_compilation.html](http://docs.cython.org/src/userguide/source_files_and_compilation.html)


Usage
-------

See simulate.py for a simple example (requires h5py for storage of coordinates and velocities)

Additional force fields may be added by inheriting from `cdef class Force` and overriding the `evaluate` method.
