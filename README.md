# MicrographModeller
Simulation of electron microscopy tilt- and frame-series, focused on modelling whole samples.

Before installation, check the `env.yaml` file. Update the cudatoolkit version in the `env.yaml` file if it does not 
match your current cudatoolkit installation. Conda will then solve the right version of cupy.
```
conda env create -f env.yaml --name mm_env
conda activate mm_env
```

WIP: repo under construction; pytom dependencies need to be removed
