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

Two options to create the conda environment:

- (1) this will later build cupy against local cuda toolkit: `conda create --name 
  micrograph_modeller python=3`

- (2) this gets a prebuild cupy and compatible cuda-toolkit: `conda create --name 
  micrograph_modeller python=3 cupy cuda-version=11.8`

Then activate the environment, clone the repository and install:

```commandline
conda activate micrograph_modeller
git clone https://github.com/McHaillet/micrograph-modeller.git
cd micrograph-modeller
python -m pip install .
```

## Tests

You can run the unittests as follows.

```commandline

```

## Running

UNDER CONSTRUCTION
