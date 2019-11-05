from setuptools import setup, Extension
import numpy as np

# To compile and install locally run "python setup.py build_ext --inplace"
# To install library to Python site-packages run "python setup.py build_ext install"

ext_modules = [
    Extension(
        'pycocotools.pycocotools._mask',
        sources=['pycocotools/pycocotools/maskApi.c', 'pycocotools/pycocotools/_mask.pyx'],
        include_dirs=[np.get_include(), 'pycocotools'],
        extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99'],
    )
]

setup(
    name='pycocotools',
    packages=['pycocotools'],
    package_dir={'pycocotools': 'pycocotools'},
    install_requires=[
        'setuptools>=18.0',
        'cython>=0.27.3',
        'matplotlib'
    ],
    version='2.0',
    ext_modules=ext_modules
)
