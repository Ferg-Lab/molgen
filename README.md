MolGen
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/molgen/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/molgen/actions?query=workflow%3ACI)
<!--[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/MolGen/branch/main/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/MolGen/branch/main)-->


This package implements GANs for generative tasks such as conditional molecular structure generation.

Getting Started
===============


Installation
------------
To use `mdn_propagator`, you will need an environment with the following packages:

* Python 3.7+
* [PyTorch](https://pytorch.org/get-started/locally/)
* [PyTorch Lightning](https://www.pytorchlightning.ai/)

For running and visualizing examples:
* [NumPy](https://numpy.org/install/)
* [MDTraj](https://www.mdtraj.org/1.9.8.dev0/installation.html)
* [NGLView](https://github.com/nglviewer/nglview#installation)

Once you have these packages installed, you can install molecool in the same environment using

```
$ pip install -e .
```

Usage
-------
Once installed, you can use the package. This example generates a synthetic trajectory of Alanine Dipeptide (ADP) in the space of the backbone dihedral angles ($\phi , \psi$). More detailed examples can be found in the `examples` directory. 


```python
from molgen.models import WGANGP
from pathlib import Path
import mdtaj as md
import torch
import numpy as np

# load data
pdb_fname = 'examples/data/alanine-dipeptide-nowater.pdb'
trj_fnames = [str(i) for i in Path('examples/data/').glob('alanine-dipeptide-*-250ns-nowater.xtc')]
trjs  = [md.load(t, top=pdb_fname).center_coordinates() for t in trj_fnames]


# process xyz coordinates and conditioning variables
xyz = list()
phi_psi = list()
for trj in trjs:
    
    t_backbone = trj.atom_slice(trj.top.select('backbone')).center_coordinates()
    
    n = trj.xyz.shape[0]
    
    _, phi = md.compute_phi(trj)
    _, psi = md.compute_psi(trj)
    
    xyz.append(torch.tensor(t_backbone.xyz.reshape(n, -1)).float())
    phi_psi.append(torch.tensor(np.concatenate((phi, psi), -1)).float())

# ininstantiate the model
model = WGANGP(xyz[0].shape[1], phi_psi[0].shape[1])

# fit the model
model.fit(xyz, phi_psi, max_epochs=25)

# Generate synthetic configurations
xyz_gen = model.generate(torch.cat(phi_psi))
xyz_gen = xyz_gen.reshape(xyz_gen.size(0), -1, 3)

# Save model checkpoint
model.save('ADP.ckpt')

# Load from checkpoint
model = WGANGP.load_from_checkpoint('ADP.ckpt')
```

### Copyright

Copyright (c) 2023, Kirill Shmilovich


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
