# SVRG_in_TNS

### General

This is a simple example code of Gradient Descent Method on Matrix Product State. I wrote it for my undergraduate thesis. This is not a formal project and it may have many bugs XD.

The tensor type is defined by Prof. B Clark at UIUC. See his [Problem Set 3](https://courses.physics.illinois.edu/phys598bkc/fa2015/hw3.html).

I developed SVRG/SCSG and Adam algorithms about Gradient Optimization on 1-dim Tensor Network (MPS).

### Usage

Please open these `*.ipynb` file via Jupyter Lab. You can install it [here](https://jupyter.org/install.html) or use the following command in [Julia](https://julialang.org/):
```julia
using Pkg; Pkg.add("IJulia")
```

After the installation, enter this repository and call up Jupyter Lab by Julia: 
```bash
$ julia -e 'using IJulia; jupyterlab(detached=true, dir=".")' 
```

### To Professor Carlson

Tests about Heisanberg model was placed in [Heisenberg.ipynb](https://github.com/mrsun-97/SVRG_in_TNS/blob/master/Heisenberg.ipynb).
