from distutils.core import setup
from distutils.extension import Extension

import numpy
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

try:
    from Cython.Distutils import build_ext
    ext_modules = [Extension("ForceFields", ["ForceFields.pyx","ForceFields.pxd"],
                include_dirs=[numpy_include],
                extra_compile_args=["-O3","-ffast-math"]),
               Extension("cIntegrator", ["cIntegrator.pyx"],
                include_dirs=[numpy_include],
                extra_compile_args=["-O3","-ffast-math"])]
                
    setup(name = 'Python ForceFields and Integrator modules',
          cmdclass = {'build_ext': build_ext},
          ext_modules = ext_modules)
                
except:
    ext_modules = [Extension("ForceFields", ["ForceFields.c"],
                include_dirs=[numpy_include],
                extra_compile_args=["-O3","-ffast-math"]),
               Extension("cIntegrator", ["cIntegrator.c"],
                include_dirs=[numpy_include],
                extra_compile_args=["-O3","-ffast-math"])]
               
    setup(
      name = 'Python ForceFields and Integrator modules',
      ext_modules = ext_modules
    )

