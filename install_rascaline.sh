#!/bin/bash
conda install -c conda-forge cxx-compiler gcc rust python=3.10

export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

export CC="$CONDA_PREFIX/bin/gcc"
export CXX="$CONDA_PREFIX/bin/g++"

export CMAKE_C_COMPILER="$CONDA_PREFIX/bin/gcc"
export CMAKE_CXX_COMPILER="$CONDA_PREFIX/bin/g++"

#purge conda cache
conda clean --all

#purge the pip cache
pip cache purge

#install plumed and pyplumed from i-pi-demo channel with crystallization enabled

# pip installs
pip install cmake numpy
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.3.0

pip install ase
pip install metatensor
pip install metatensor-core
pip install metatensor-operations
pip install metatensor-torch

pip install git+https://github.com/Luthaf/rascaline
pip install --extra-index-url https://download.pytorch.org/whl/cpu git+https://github.com/luthaf/rascaline#subdirectory=python/rascaline-torch

# modifies the i_pi drivers
