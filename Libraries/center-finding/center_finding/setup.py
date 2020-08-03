from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
	ext_modules=cythonize("cython_center_finding_tools.pyx"),
	include_dirs=[numpy.get_include()]
)
