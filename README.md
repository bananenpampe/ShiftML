# ShiftML
Model to predict chemcial shieldings.

To install the dependencies create a fresh conda environment and run the following command:

```bash
conda create --name shiftML

conda activate shiftML

bash install_rascaline.sh
```

If your mac/pc is too old some of the gcc/cxx compilers are not available, you need to install the c compilers yourself.

on a mac you can do this:
```
conda install cmake
brew install cmake
brew install gcc
```
change the environemnt variables to:

```
export CC="/usr/bin/gcc"
export CXX="/usr/bin/g++"

export CMAKE_C_COMPILER="/usr/bin/gcc"
export CMAKE_CXX_COMPILER="/usr/bin/g++"
```

and potentially you also need to specify another torch version in the pip install.

```
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.2.0
```

