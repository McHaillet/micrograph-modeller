# MicrographModeller
Simulation of electron microscopy tilt- and frame-series, focused on modelling whole samples.

## Dependencies

Micrograph-modeller is a python(>=3.8) package that can be installed. The base of the code runs on CPU's due to the 
large memory requirements for the tomography models. For sampling atoms to electrostatic potential a part of the 
code can be accelerated with GPU's. You will need to install cupy and CUDA toolkit to access it (see below).

To add hydrogens and remove water molecules from pdb/cif files I call chimerax from the command line. So for the 
full functionality you need to have chimerax installed.

## Installation

I assume you have miniconda3 installed.

First create an environment with cupy in case you want to use the GPU acceleration. Conda will also automatically 
install a CUDA toolkit version compatible with cupy in this environment. You can specify more recent versions 
of cupy, but just keep in mind everything has been developed with 10.6.

```commandline
conda create --name micrograph_modeller python=3.8 cupy=10.6.0
```

If you need a specific version of cudatoolkit you can specify it as well.

```commandline
conda create --name micrograph_modeller python=3.8 cupy=10.6.0 cudatoolkit=11.6
```

You can also install a wheel of cupy via pip that is build against an installed versions of CUDA toolkit on your 
system. This can be preferred because CUDA toolkit takes up a lot of disk space.

```commandline
conda create --name micrograph_modeller python=3.8
conda activate micrograph_modeller
python -m pip install cupy-cuda116=10.6.0
```

If you did not do so already, activate the environment.

```commandline
conda activate micrograph_modeller
```

Clone the repository.

```commandline
git clone https://github.com/McHaillet/micrograph-modeller.git
```

And then **install micrograph-modeller**.

```commandline
cd micrograph-modeller
python -m pip install .
```

## Tests

You can run the unittests as follows.

```commandline

```

## Running

UNDER CONSTRUCTION
