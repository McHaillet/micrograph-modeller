#!/usr/bin/env python

import argparse
import os
import sys
import micrographmodeller as mm


if __name__ == '__main__':
    # SOME OF THE SCRIPT FUNCTIONALITY DEPEDNS ON CHIMERA (1.12; not tested with chimeraX), PDB2PQR (modified), APBS
    parser = argparse.ArgumentParser(description='Calculate electrostatic potential from protein structure file. This '
                                                 'script will automatically attempt to call chimera to add '
                                                 'hydrogens to a pdb. For improved electrostatic potential its '
                                                 'possible to run in combination with PDB2PQR and APBS. But this '
                                                 'might only matter at very small pixel sizes. Behaviour with APBS is '
                                                 'also not fully tested. -- Marten Chaillet (@McHaillet)')
    parser.add_argument('-f', '--file', type=str, required=True,
                        help='File path with protein structure, either pdb or cif.')
    parser.add_argument('-d', '--destination', type=str, required=False, default='./',
                        help='Folder to store the files produced by potential.py. Default is current folder.')
    parser.add_argument('--skip-edit', action='store_true', default=False, required=False,
                        help='Whether to call schrodinger/pymol2 module for some structure modification: '
                             'adding hydrogens, removing water molecules, and adding crystal symmetry.')
    parser.add_argument('-s', '--spacing', type=float, required=False, default=1.,
                        help='The size of the voxels of the output volume. 1A by default.')
    parser.add_argument('-n', '--oversampling', type=int, required=False, nargs='?', const=2, default=1,
                        help='n times pixel size oversampling. If argument is provided without value, will '
                             'oversample 2 times.')
    parser.add_argument('-b', '--binning', type=int, required=False, default=1,
                        help='Number of times to bin. Additional storage of binned volume.')
    parser.add_argument('-x', '--exclude-solvent', type=str, required=False, choices=['gaussian', 'masking'],
                        help='Whether to exclude solvent around each atom as a correction of the potential, '
                             'either "gaussian" or "masking".')
    parser.add_argument('-p', '--solvent-potential', type=float, required=False, default=mm.physics.V_WATER,
                        help=f'Value for the solvent potential. By default amorphous ice, {mm.physics.V_WATER} V.')
    parser.add_argument('-a', '--absorption-contrast', action='store_true', default=False, required=False,
                        help='Whether to generate imaginary part of molecule potential, can only be done if solvent'
                             'is excluded.')
    parser.add_argument('-v', '--voltage', type=float, required=False, default=300,
                        help='Value for the electron acceleration voltage. Needed for calculating the inelastic mean '
                             'free path in case of absorption contrast calculation. By default 300 (keV).')
    parser.add_argument('-c', '--cores', type=int, required=False, default=1,
                        help='Number of cpu cores to use for the calculation.')
    parser.add_argument('-g', '--gpu-id', type=int, required=False,
                        help='GPU index to run the program on.')

    args = parser.parse_args()
    # check if io locations are valid
    if not os.path.exists(args.file):
        print('Input file does not exist, exiting...')
        sys.exit(0)
    if not os.path.exists(args.destination):
        print('Destination for writing files does not exist, exiting...')
        sys.exit(0)

    mm.potential.wrapper(args.file, args.destination, args.spacing, skip_structure_edit=args.skip_edit,
                         oversampling=args.oversampling, binning=args.binning,
                         solvent_exclusion=args.exclude_solvent, solvent_potential=args.solvent_potential,
                         absorption_contrast=args.absorption_contrast, voltage=args.voltage * 1e3, cores=args.cores,
                         gpu_id=args.gpu_id)
