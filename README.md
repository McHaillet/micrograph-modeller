# MicrographModeller
Simulation of electron microscopy tilt- and frame-series, focused on modelling whole 
samples. This is the code that was used to generate the [SHREC'21 dataset](https://dataverse.nl/dataset.xhtml;jsessionid=d97253a9bab755b1bd31d843252e?persistentId=doi%3A10.34894%2FXRTJMA&version=&q=&fileTypeGroupFacet=%22Archive%22&fileAccess=Public).

The code is not well optimized, and uses a lot of CPU memory. It does generate 
fairly accurate simulations as it also models amplitude contrast from inelastic 
interactions (where it assumes a zero-loss filtered image, corresponding to a modern 
post-column energy filter). It can also generate deformed vesicles, in contrast to 
SHREC'21 where it only generated ellipsoidal vesicles (no membrane protein embedding 
though).

The simulator is heavily based on InSilicoTEM, by Vulovic et al., (2013): 
https://doi.org/10.1016/j.jsb.2013.05.008 .

[//]: # (## Related work)

[//]: # ()
[//]: # (There are many cool new simulators at the moment, just to list a few and their )

[//]: # (strong points &#40;as far as I understand&#41;:)

[//]: # ()
[//]: # (- TEM simulator, early work simulator, but tomotwin wrote some neat wrappers around it:)

[//]: # (- this one from lorenz)

[//]: # (- CisTem simulator => seems most physically accurate by far)

[//]: # (- ParaKeet, has a cool ice noise model)

[//]: # (- CryoSim, from swulius lab)

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
cd tests
python -m unittest discover
```

## Running

UNDER CONSTRUCTION
