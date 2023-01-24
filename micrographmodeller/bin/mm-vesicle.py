#!/usr/bin/env python

import time
import argparse
import os
import sys
import micrographmodeller as mm
from importlib import resources as importlib_resources


if __name__ == '__main__':
    # more lipid bilayer boxes: https://people.ucalgary.ca/~tieleman/download.html
    start = time.time()

    parser = argparse.ArgumentParser(description='Generate a vesicle. Script will create a triangular mesh framework '
                                                 'on a random ellipsoidal shape of average radius (-r). Lipids will '
                                                 'be sampled from a MD-equilibrated lipid bilayer pdb structure with '
                                                 'periodic boundaries. -- Marten Chaillet (@McHaillet)')
    parser.add_argument('-r', '--radius', type=float, required=True,
                        help='Average radius of the vesicle in nm.')
    parser.add_argument('-s', '--spacing', type=float, required=False, default=1,
                        help='Voxel spacing to sample electrostatic potential on in A.')
    parser.add_argument('-d', '--destination', type=str, required=False, default='./',
                        help='Folder to write output to, default is current folder.')
    parser.add_argument('-m', '--membrane-pdb', type=str, required=False,
                        help='Membrane model file (make sure waters are removed), default is dppc128_dehydrated.pdb. '
                             'Which is included in the package data. You can also provide your own by specifying the '
                             'path to a .pdb file. The bilayer needs to be flat and preferably have periodic '
                             'boundaries. examples: https://people.ucalgary.ca/~tieleman/download.html.')
    parser.add_argument('-x', '--exclude-solvent', type=str, required=False, choices=['gaussian', 'masking'],
                        help='Whether to exclude solvent around each atom as a correction of the potential, '
                             'either "gaussian" or "masking".')
    parser.add_argument('-p', '--solvent-potential', type=float, required=False, default=mm.physics.V_WATER,
                        help=f'Value for the solvent potential. By default amorphous ice, {mm.physics.V_WATER} V.')
    parser.add_argument('-v', '--voltage', type=float, required=False, default=300,
                        help='Value for the electron acceleration voltage. Needed for calculating the inelastic mean '
                             'free path in case of absorption contrast calculation. By default 300 (keV).')
    parser.add_argument('-c', '--cores', type=int, required=False, default=1,
                        help='Number of cpu cores to use for the calculation.')
    parser.add_argument('-g', '--gpu-id', type=int, required=False,
                        help='GPU index to run the program on.')

    args = parser.parse_args()
    # check if io locations are valid
    if args.membrane_pdb is None:
        with importlib_resources.path(mm, 'membrane_models/dppc128_dehydrated.pdb') as path:
            args.membrane_pdb = str(path)
    elif not os.path.exists(args.membrane_pdb):
        print('Input file does not exist, exiting...')
        sys.exit(0)

    if not os.path.exists(args.destination):
        print('Destination for writing files does not exist, exiting...')
        sys.exit(0)

    # find good number of points to sample: a 23nm radius vesicle is good with 100 points
    size_factor = args.radius / 23
    sampling_points = int(100 * size_factor**2.2)  # number of points
    alpha = 2000 * size_factor

    vesicle = mm.membrane.Vesicle(args.radius * 10, args.spacing)  # radius in A
    vesicle.sample_ellipsoid_point_cloud(sampling_points)
    vesicle.equilibrate_point_cloud(maxiter=10000, factor=0.1)
    vesicle.deform(args.radius / 4)
    vesicle.generate_framework(alpha)
    structure_tuple = vesicle.sample_membrane(args.membrane_pdb, cores=args.cores)

    # sample the atoms to voxels
    ep = mm.potential.ElectrostaticPotential(structure_tuple, solvent_exclusion=args.exclude_solvent,
                                             solvent_potential=args.solvent_potential, absorption_contrast=True,
                                             voltage=args.voltage * 1e3, protein_density=mm.physics.PROTEIN_DENSITY,
                                             molecular_weight=mm.physics.PROTEIN_MW)
    potential = ep.sample_to_box(voxel_size=args.spacing, center_coordinates_in_box=True, overhang=20,
                                 gpu_id=args.gpu_id, cores=args.cores)

    # filter and write
    potential = mm.support.reduce_resolution_real(potential, args.spacing, 2 * args.spacing)

    name = 'bilayer'  # double values to get diameters of ellipsoid
    size = f'{vesicle.radii[0] * 2 / 10:.0f}x{vesicle.radii[1] * 2 / 10:.0f}x{vesicle.radii[2] * 2 / 10:.0f}nm'

    mm.support.write_mrc(os.path.join(args.destination, f'{name}_{size}_{args.spacing:.2f}A_solvent-4.530V_real.mrc'),
                         potential.real, args.spacing)
    mm.support.write_mrc(os.path.join(args.destination,
                                   f'{name}_{size}_{args.spacing:.2f}A_solvent-4.530V_imag_300V.mrc'),
                         potential.imag, args.spacing)

    end = time.time()

    print('\n Time elapsed: ', end-start, '\n')
