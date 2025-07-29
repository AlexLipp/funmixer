import setuptools

from Cython.Build import cythonize
from setuptools import Extension
import numpy

__version__ = "0.1.1"

ext_modules = cythonize(
    Extension(
        "funmixer.flow_acc_cfuncs",
        ["funmixer/flow_acc_cfuncs.pyx"],
        language="c++",  # Use C++ compiler
    )
)

setuptools.setup(
    name="funmixer",
    version=__version__,
    description="Convex unmixing of fluid networks",
    url="https://github.com/AlexLipp/funmixer",
    author="Richard Barnes & Alex Lipp",
    author_email="rijard.barnes@gmail.com; alex@lipp.org.uk",
    license="GPLv3",
    packages=setuptools.find_packages(),
    ext_modules=ext_modules,
    keywords="GIS hydrology raster networks",
    python_requires=">= 3.8, <4",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Other Audience",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    include_dirs=[numpy.get_include()],
    install_requires=[
        "cvxpy",
        "cython",
        "gdal",
        "hypothesis",
        "matplotlib",
        "networkx",
        "numpy",
        "pandas",
        "pybind11",
        "pytest",
        "pygraphviz",
        "tqdm",
    ],
)
