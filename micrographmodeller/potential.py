import multiprocessing as mp
import numpy as np
import os
import sys
import scipy.ndimage as ndimage
import micrographmodeller.physics as physics
import micrographmodeller.support as support
import micrographmodeller.utils as utils
import micrographmodeller.pdbs as pdbs


# // N should be len(atoms) // 3
# // <<<N/TPB, TPB>>>
#
# // #include "math.h"  => can probably be skipped as cupy math.h probably includes erf()

iasa_integrate_text = """
#define M_PI 3.141592654

extern "C" __global__ void iasa_integrate(
                    float3 *atoms, unsigned char *elements, float *b_factors, float *occupancies, 
                    float *potential, float *solvent, float *scattering_factors, unsigned int *potential_dims,
                    float *displaced_volume, float voxel_size, unsigned int n_atoms, unsigned char exclude_solvent) 
{
    // exclude_solvent is used a Boolean value
    
    // get the atom index                                                                                                                                      
    unsigned int i = (blockIdx.x*(blockDim.x) + threadIdx.x); // correct for each atom having 3 coordinates
        
    if (i < n_atoms) {
    
        // make atoms a float3
        // potential_dims can also be a float3
        
        unsigned int j, l, m, n, potent_idx;
        int3 ind_min, ind_max;
        float atom_voxel_pot, sqrt_b, pi2_sqrt_b, pi2, sqrt_pi, factor3;
        float integral_x, integral_y, integral_z, integral_voxel;
        float3 voxel_bound_min, voxel_bound_max;
    
        // get all the atom information
        //atom.x = atoms[(i * 3) + 0];
        //atom.y = atoms[(i * 3) + 1];
        //atom.z = atoms[(i * 3) + 2];
        int elem = elements[i];  // its much easier if elements are ints, because dict wont work in C
        float b_factor = b_factors[i];
        float occupancy = occupancies[i];
        
        // scattering factors for this atom
        float *a = &scattering_factors[elem * 10];
        float *b = &scattering_factors[elem * 10 + 5];
        
        // get the radius of the volume displacement
        float r0 = cbrt(displaced_volume[elem] / sqrt(M_PI * M_PI * M_PI));
        
        // find the max radius over all gaussians with a certain cutoff
        float r2 = 0;
        for (j = 0; j < 5; j++) {
            r2 = max(r2, 15 / (4 * M_PI * M_PI / b[j]));
        };
        float r = sqrt(r2 / 3);
        
        // set indices in potential box
        ind_min.x = (int)floor((atoms[i].x - r) / voxel_size); // use floor and int casting for explicitness
        ind_max.x = (int)floor((atoms[i].x + r) / voxel_size);
        ind_min.y = (int)floor((atoms[i].y - r) / voxel_size);
        ind_max.y = (int)floor((atoms[i].y + r) / voxel_size);
        ind_min.z = (int)floor((atoms[i].z - r) / voxel_size);
        ind_max.z = (int)floor((atoms[i].z + r) / voxel_size);
        
        // enforce ind_min can never be smaller than 0 and ind_max can never be larger than potential_dims
        // ind min should also always be smaller than the dims...
        ind_min.x = max(ind_min.x, 0);
        ind_min.y = max(ind_min.y, 0);
        ind_min.z = max(ind_min.z, 0);
        ind_max.x = min(ind_max.x, potential_dims[0] - 1);
        ind_max.y = min(ind_max.y, potential_dims[1] - 1);
        ind_max.z = min(ind_max.z, potential_dims[2] - 1);
        
        // break if ind_min larger than box or ind_max smaller than 0
        if ((ind_min.x < potential_dims[0] - 1) && (ind_min.y < potential_dims[1] - 1) && 
            (ind_min.z < potential_dims[2] - 1) && (ind_max.x > 0) && (ind_max.y > 0) && 
            (ind_max.z > 0)) {        
            
            // precalc sqrt of pi
            sqrt_pi = sqrt(M_PI);
            // two times pi
            pi2 = 2 * M_PI;
            // loop over coordinates where this atom is present
            for (l = ind_min.x; l < (ind_max.x + 1); l++) {
            
                voxel_bound_min.x = l * voxel_size - atoms[i].x;
                voxel_bound_max.x = (l + 1) * voxel_size - atoms[i].x;
                
                for (m = ind_min.y; m < (ind_max.y + 1); m++) {
                
                    voxel_bound_min.y = m * voxel_size - atoms[i].y;
                    voxel_bound_max.y = (m + 1) * voxel_size - atoms[i].y;
                    
                    for (n = ind_min.z; n < (ind_max.z + 1); n++) {
                    
                        voxel_bound_min.z = n * voxel_size - atoms[i].z;
                        voxel_bound_max.z = (n + 1) * voxel_size - atoms[i].z;
                        
                        // initialize to zero for this voxel
                        atom_voxel_pot = 0;
                        
                        for (j = 0; j < 5; j++) {
                            sqrt_b = sqrt(b[j]);
                            pi2_sqrt_b = pi2 / sqrt_b;
                            factor3 = powf(sqrt_b / (4 * sqrt_pi), 3);
                            
                            integral_x = (erf(voxel_bound_max.x * pi2_sqrt_b) - erf(voxel_bound_min.x * pi2_sqrt_b));
                            integral_y = (erf(voxel_bound_max.y * pi2_sqrt_b) - erf(voxel_bound_min.y * pi2_sqrt_b));
                            integral_z = (erf(voxel_bound_max.z * pi2_sqrt_b) - erf(voxel_bound_min.z * pi2_sqrt_b));
                            integral_voxel = integral_x * integral_y * integral_z * factor3;
                            
                            atom_voxel_pot += (a[j] / powf(b[j], (float)3 / 2)) * integral_voxel;
                        };
                        
                        potent_idx = l * potential_dims[1] * potential_dims[2] + m * potential_dims[2] + n;
                        atomicAdd( potential + potent_idx, atom_voxel_pot );
                        
                        if (exclude_solvent == 1) {
                            factor3 = powf(sqrt_pi * r0 / 2, 3);
                            
                            integral_x = erf(voxel_bound_max.x / r0) - erf(voxel_bound_min.x / r0);
                            integral_y = erf(voxel_bound_max.y / r0) - erf(voxel_bound_min.y / r0);
                            integral_z = erf(voxel_bound_max.z / r0) - erf(voxel_bound_min.z / r0);
                            
                            atomicAdd( solvent + potent_idx, factor3 * integral_x * integral_y * integral_z );
                        
                        };
                    };
                };
            };
        };
    };
};
"""


def create_gold_marker(
    voxel_size,
    solvent_potential,
    oversampling=1,
    solvent_factor=1.0,
    imaginary=False,
    voltage=300e3,
):
    """
    From Rahman 2018 (International Journal of Biosensors and Bioelectronics).
    Volume of unit cell gold is 0.0679 nm^3 with 4 atoms per unit cell.
    Volume of gold bead is 4/3 pi r^3.

    @param voxel_size: voxel size of the box where gold marker is generated, in A
    @type  voxel_size: L{float}
    @param solvent_potential: solvent background potential
    @type  solvent_potential: L{float}
    @param oversampling: number of times to oversample the voxel size for more accurate generation
    @type  oversampling: L{int}
    @param solvent_factor: factor for denser solvent
    @type  solvent_factor: L{float}
    @param imaginary: flag for generating imaginary part of the potential
    @type  imaginary: L{bool}
    @param voltage: voltage of electron beam in eV, default 300E3
    @type  voltage: L{float}

    @return: if imaginary is True, return tuple (real, imaginary), if false return only real. boxes real and imag are
    3d arrays.
    @rtype: L{tuple} -> (L{np.ndarray},) * 2 or L{np.ndarray}

    @author: Marten Chaillet
    """
    assert (type(oversampling) is int) and (oversampling >= 1), print(
        "Stop gold marker creation oversampling factor" " is not a positive integer."
    )

    # select a random size for the gold marker in nm
    diameter = np.random.uniform(low=4.0, high=10.0)

    # constants
    unit_cell_volume = 0.0679  # nm^3
    atoms_per_unit_cell = 4
    C = (
        2
        * np.pi
        * physics.constants["h_bar"] ** 2
        / (physics.constants["el"] * physics.constants["me"])
        * 1e20
    )  # nm^2
    voxel_size_nm = (voxel_size / 10) / oversampling
    voxel_volume = voxel_size_nm**3

    # dimension of gold box, always add 5 nm to the sides
    dimension = int(np.ceil(diameter / voxel_size_nm)) * 3
    # sigma half of radius?
    r = 0.8 * (
        (diameter * 0.5) / voxel_size_nm
    )  # fraction of radius due to extension with exponential smoothing
    ellipse = True
    if ellipse:
        r2 = r * np.random.uniform(0.8, 1.2)
        r3 = r * np.random.uniform(0.8, 1.2)
        bead = support.create_ellipsoid(dimension, r, r2, r3, smooth=2)
    else:
        bead = support.create_sphere((dimension,) * 3, radius=r)

    bead *= support.add_correlated_noise(
        int(r * 0.75), dimension
    ) * support.add_correlated_noise(int(r * 0.25), dimension)
    # SIGMA DEPENDENT ON VOXEL SIZE
    # rounded_sphere = gaussian3d(sphere, sigma=(1 * 0.25 / voxel_size_nm))
    bead[bead < 0.9] = 0  # remove too small values
    # add random noise to gold particle to prevent perfect CTF ringing around the particle.
    # random noise also dependent on voxel size maybe?
    # rounded_sphere = (rounded_sphere > 0) * (rounded_sphere * np.random.normal(1, 0.3, rounded_sphere.shape))
    # rounded_sphere[rounded_sphere < 0] = 0

    if imaginary:
        solvent_amplitude = (
            physics.potential_amplitude(
                physics.AMORPHOUS_ICE_DENSITY, physics.WATER_MW, voltage
            )
            * solvent_factor
        )
        gold_amplitude = physics.potential_amplitude(
            physics.GOLD_DENSITY, physics.GOLD_MW, voltage
        )
        gold_imaginary = bead * (gold_amplitude - solvent_amplitude)
        # filter and bin
        gold_imaginary = ndimage.zoom(
            support.reduce_resolution_real(gold_imaginary, 1, 2 * oversampling),
            1 / oversampling,
            order=3,
        )

    # values transformed to occupied volume per voxel from 1 nm**3 per voxel to actual voxel size
    solvent_correction = bead * (solvent_potential * solvent_factor)
    unit_cells_per_voxel = bead * voxel_volume / unit_cell_volume
    gold_atoms = unit_cells_per_voxel * atoms_per_unit_cell

    # interaction potential
    gold_scattering_factors = np.array(physics.scattering_factors["AU"]["g"])
    # gold_scattering_factors[0:5].sum() == 10.57
    # C and scattering factor are in A units thus divided by 1000 A^3 = 1 nm^3 to convert
    gold_potential = (
        gold_atoms * gold_scattering_factors[0:5].sum() * C / voxel_volume / 1000
    )
    gold_real = gold_potential - solvent_correction
    # filter and bin
    gold_real = ndimage.zoom(
        support.reduce_resolution_real(gold_real, 1, 2 * oversampling),
        1 / oversampling,
        order=3,
    )

    if imaginary:
        return gold_real, gold_imaginary
    else:
        return gold_real


def split_data(data_length, cores):
    if data_length == cores:
        indices = []
        for i in range(cores):
            indices.append((i, i + 1))
        return indices
    elif data_length > cores:
        indices = [None] * cores
        n, N = 0, data_length
        for i in range(cores):
            l = N // cores + (N % cores > i)
            indices[i] = (n, n + l)
            n += l
        return indices
    else:
        indices = []
        for i in range(data_length):
            indices.append((i, i + 1))
        return indices


def init(shared_data_, potential_shared_, solvent_shared_):
    global shared_data
    shared_data = (
        shared_data_  # must be inherited, not passed as an argument to workers
    )
    global potential_shared
    global solvent_shared
    potential_shared = potential_shared_
    solvent_shared = solvent_shared_


def tonumpyarray(mp_arr, shape, dt):
    return np.frombuffer(mp_arr.get_obj(), dtype=dt).reshape(shape)


def parallel_integrate(index, size, solvent_exclusion, voxel_size, dtype):
    from scipy.special import erf

    # return np.ndarray view of mp.Array
    potential = tonumpyarray(potential_shared, size, dtype)
    solvent = tonumpyarray(solvent_shared, size, dtype)

    print(
        f" --- process {mp.current_process().name} calculating {index[1]-index[0]} atoms"
    )

    for i in range(index[0], index[1]):
        # order = x y z e b o
        x, y, z, element, b_factor, occupancy = (
            shared_data[0][i],
            shared_data[1][i],
            shared_data[2][i],
            shared_data[3][i],
            shared_data[4][i],
            shared_data[5][i],
        )

        # atom type
        atom = element.upper()
        # atom center
        rc = [x, y, z]

        sf = np.array(physics.scattering_factors[atom]["g"])
        a = sf[0:5]
        b = sf[5:10]

        # b += (b_factor) # units in A

        if atom in list(physics.volume_displaced):
            r_0 = np.cbrt(physics.volume_displaced[atom] / (np.pi ** (3 / 2)))
        else:  # If not H,C,O,N we assume the same volume displacement as for carbon
            r_0 = np.cbrt(physics.volume_displaced["C"] / (np.pi ** (3 / 2)))

        r2 = 0  # it was 15 / (1 / r_0 ** 2) before but this gives very high values for carbon
        for j in range(5):
            # Find the max radius over all gaussians (assuming symmetrical potential to 4.5 sigma truncation
            # (corresponds to 10).
            r2 = np.maximum(r2, 15 / (4 * np.pi**2 / b[j]))
        # Radius of gaussian sphere
        r = np.sqrt(r2 / 3)

        ind_min = [
            max(int((c - r) / voxel_size), 0) for c in rc
        ]  # Smallest index to contain relevant potential x,y,z
        ind_max = [
            min(int((c + r) / voxel_size), s - 1) for (c, s) in zip(rc, size)
        ]  # largest relevant index

        # only add if valid atom box
        if (
            (ind_min[0] < size[0] - 1)
            and (ind_min[1] < size[1] - 1)
            and (ind_min[2] < size[2] - 1)
            and (ind_max[0] > 0)
            and (ind_max[1] > 0)
            and (ind_max[2] > 0)
        ):

            # Explicit real space coordinates for the max and min boundary of each voxel
            x_min_bound = np.arange(ind_min[0], ind_max[0] + 1, 1) * voxel_size - rc[0]
            x_max_bound = (
                np.arange(ind_min[0] + 1, ind_max[0] + 2, 1) * voxel_size - rc[0]
            )
            y_min_bound = np.arange(ind_min[1], ind_max[1] + 1, 1) * voxel_size - rc[1]
            y_max_bound = (
                np.arange(ind_min[1] + 1, ind_max[1] + 2, 1) * voxel_size - rc[1]
            )
            z_min_bound = np.arange(ind_min[2], ind_max[2] + 1, 1) * voxel_size - rc[2]
            z_max_bound = (
                np.arange(ind_min[2] + 1, ind_max[2] + 2, 1) * voxel_size - rc[2]
            )

            atom_potential = 0

            for j in range(5):
                sqrt_b = np.sqrt(b[j])  # calculate only once
                # Difference of error function == integrate over Gaussian
                int_x = (
                    sqrt_b
                    / (4 * np.sqrt(np.pi))
                    * (
                        erf(x_max_bound * 2 * np.pi / sqrt_b)
                        - erf(x_min_bound * 2 * np.pi / sqrt_b)
                    )
                )
                x_matrix = np.tile(
                    int_x[:, np.newaxis, np.newaxis],
                    [1, ind_max[1] - ind_min[1] + 1, ind_max[2] - ind_min[2] + 1],
                )
                int_y = (
                    sqrt_b
                    / (4 * np.sqrt(np.pi))
                    * (
                        erf(y_max_bound * 2 * np.pi / sqrt_b)
                        - erf(y_min_bound * 2 * np.pi / sqrt_b)
                    )
                )
                y_matrix = np.tile(
                    int_y[np.newaxis, :, np.newaxis],
                    [ind_max[0] - ind_min[0] + 1, 1, ind_max[2] - ind_min[2] + 1],
                )
                int_z = (
                    sqrt_b
                    / (4 * np.sqrt(np.pi))
                    * (
                        erf(z_max_bound * 2 * np.pi / sqrt_b)
                        - erf(z_min_bound * 2 * np.pi / sqrt_b)
                    )
                )
                z_matrix = np.tile(
                    int_z[np.newaxis, np.newaxis, :],
                    [ind_max[0] - ind_min[0] + 1, ind_max[1] - ind_min[1] + 1, 1],
                )

                atom_potential += (
                    a[j] / b[j] ** (3 / 2) * x_matrix * y_matrix * z_matrix
                )

            potential[
                ind_min[0] : ind_max[0] + 1,
                ind_min[1] : ind_max[1] + 1,
                ind_min[2] : ind_max[2] + 1,
            ] += atom_potential

            if solvent_exclusion == "gaussian":
                # excluded solvent potential
                int_x = (
                    np.sqrt(np.pi)
                    * r_0
                    / 2
                    * (erf(x_max_bound / r_0) - erf(x_min_bound / r_0))
                )
                x_matrix = np.tile(
                    int_x[:, np.newaxis, np.newaxis],
                    [1, ind_max[1] - ind_min[1] + 1, ind_max[2] - ind_min[2] + 1],
                )
                int_y = (
                    np.sqrt(np.pi)
                    * r_0
                    / 2
                    * (erf(y_max_bound / r_0) - erf(y_min_bound / r_0))
                )
                y_matrix = np.tile(
                    int_y[np.newaxis, :, np.newaxis],
                    [ind_max[0] - ind_min[0] + 1, 1, ind_max[2] - ind_min[2] + 1],
                )
                int_z = (
                    np.sqrt(np.pi)
                    * r_0
                    / 2
                    * (erf(z_max_bound / r_0) - erf(z_min_bound / r_0))
                )
                z_matrix = np.tile(
                    int_z[np.newaxis, np.newaxis, :],
                    [ind_max[0] - ind_min[0] + 1, ind_max[1] - ind_min[1] + 1, 1],
                )

                solvent[
                    ind_min[0] : ind_max[0] + 1,
                    ind_min[1] : ind_max[1] + 1,
                    ind_min[2] : ind_max[2] + 1,
                ] += (
                    x_matrix * y_matrix * z_matrix
                )

    print(f" --- process {mp.current_process().name} finished")


def sample_iasa_cpu(
    structure_tuple, box_dimensions, voxel_size, cores, solvent_exclusion=None
):
    from functools import partial, reduce
    from contextlib import closing
    import operator
    import ctypes

    x_coordinates, y_coordinates, z_coordinates, elements, b_factors, occupancies = (
        structure_tuple
    )

    # split the data into fractions over the nodes
    indices = split_data(
        len(x_coordinates), cores
    )  # adjust nodes in case n_atoms is smaller than n_cores

    x_shared, y_shared, z_shared = (
        mp.Array("d", x_coordinates, lock=False),
        mp.Array("d", y_coordinates, lock=False),
        mp.Array("d", z_coordinates, lock=False),
    )
    b_shared, o_shared = mp.Array("d", b_factors, lock=False), mp.Array(
        "d", occupancies, lock=False
    )
    e_shared = mp.Array(ctypes.c_wchar_p, elements, lock=False)

    print(
        f"Number of atoms to go over is {len(x_coordinates)} spread over {len(indices)} processes"
    )

    # create shared arrays
    potential_shared = mp.Array(ctypes.c_float, reduce(operator.mul, box_dimensions))
    solvent_shared = mp.Array(ctypes.c_float, reduce(operator.mul, box_dimensions))
    # initialize them via numpy
    dtype = np.float32
    potential, solvent = tonumpyarray(
        potential_shared, box_dimensions, dtype
    ), tonumpyarray(solvent_shared, box_dimensions, dtype)
    potential[:], solvent[:] = np.zeros(box_dimensions, dtype=dtype), np.zeros(
        box_dimensions, dtype=dtype
    )

    with closing(
        mp.Pool(
            len(indices),
            initializer=init,
            initargs=(
                (x_shared, y_shared, z_shared, e_shared, b_shared, o_shared),
                potential_shared,
                solvent_shared,
            ),
        )
    ) as p:
        p.map_async(
            partial(
                parallel_integrate,
                size=box_dimensions,
                solvent_exclusion=solvent_exclusion,
                voxel_size=voxel_size,
                dtype=dtype,
            ),
            indices,
        )
    p.join()
    return potential, solvent


def sample_iasa_gpu(
    structure_tuple, box_dimensions, voxel_size, gpu_device, solvent_exclusion=None
):
    # cp set device ...
    cp, _ = utils.get_array_module_from_device(gpu_device)

    # unpack atom info
    x_coordinates, y_coordinates, z_coordinates, elements, b_factors, occupancies = (
        structure_tuple
    )

    # setup easier gpu indexing for scattering factors
    scattering = cp.array(
        [v["g"] for k, v in physics.scattering_factors.items()], dtype=cp.float32
    )
    map_element_to_id = {k: i for i, k in enumerate(physics.scattering_factors.keys())}

    # setup gpu indexing for solvent displacement
    displacement = cp.array(
        [
            (
                physics.volume_displaced[k]
                if k in physics.volume_displaced.keys()
                else physics.volume_displaced["C"]
            )
            for k in map_element_to_id.keys()
        ],
        dtype=cp.float32,
    )

    # find dimensions
    n_atoms = len(x_coordinates)

    # initiate the final volume
    sz_potential_gpu = cp.array(box_dimensions, dtype=cp.uint32)
    potential = cp.zeros(box_dimensions, dtype=cp.float32)
    if solvent_exclusion == "gaussian":
        solvent = cp.zeros(box_dimensions, dtype=cp.float32)
    else:
        solvent = cp.array([0], dtype=cp.float32)

    # print to user the number of atoms in system
    print(f"Number of atoms to go over is {n_atoms}")

    # create kernel
    iasa_integrate = cp.RawKernel(code=iasa_integrate_text, name="iasa_integrate")

    # move the selected atoms to gpu!
    atoms = cp.ascontiguousarray(
        cp.array([x_coordinates, y_coordinates, z_coordinates]).T, dtype=cp.float32
    )
    n_atoms_iter = atoms.shape[0]
    elements = cp.array(
        [map_element_to_id[e.upper()] for e in elements], dtype=cp.uint8
    )
    b_factors = cp.array(b_factors, dtype=cp.float32)
    occupancies = cp.array(occupancies, dtype=cp.float32)

    # threads and blocks
    n_threads = 1024
    n_blocks = int(cp.ceil(n_atoms / n_threads).get())

    # call gpu integration either with or without gaussian solvent exclusion
    if solvent_exclusion == "gaussian":
        iasa_integrate(
            (
                n_blocks,
                1,
                1,
            ),
            (n_threads, 1, 1),
            (
                atoms,
                elements,
                b_factors,
                occupancies,
                potential,
                solvent,
                scattering,
                sz_potential_gpu,
                displacement,
                cp.float32(voxel_size),
                cp.uint32(n_atoms_iter),
                cp.uint8(1),
            ),
        )
        # last argument is a Boolean for using, or not using, solvent exclusion
    else:
        iasa_integrate(
            (
                n_blocks,
                1,
                1,
            ),
            (n_threads, 1, 1),
            (
                atoms,
                elements,
                b_factors,
                occupancies,
                potential,
                solvent,
                scattering,
                sz_potential_gpu,
                displacement,
                cp.float32(voxel_size),
                cp.uint32(n_atoms_iter),
                cp.uint8(0),
            ),
        )

    return potential.get(), solvent.get()


class ElectrostaticPotential:
    def __init__(
        self,
        data,
        solvent_exclusion=None,
        solvent_potential=physics.V_WATER,
        absorption_contrast=False,
        voltage=300e3,
        protein_density=physics.PROTEIN_DENSITY,
        molecular_weight=physics.PROTEIN_MW,
    ):
        if isinstance(data, tuple):
            pass
        elif isinstance(data, str) and (data.endswith(".pdb") or data.endswith(".cif")):
            data = pdbs.read_structure(data)
        else:
            print("invalid input")
            sys.exit(0)

        # this duplicates the data ... not great
        (
            self.x_coordinates,
            self.y_coordinates,
            self.z_coordinates,
            self.elements,
            self.b_factors,
            self.occupancies,
        ) = map(np.array, data)

        self.x_limit = (self.x_coordinates.min(), self.x_coordinates.max())
        self.y_limit = (self.y_coordinates.min(), self.y_coordinates.max())
        self.z_limit = (self.z_coordinates.min(), self.z_coordinates.max())

        self.solvent_exclusion = solvent_exclusion
        self.solvent_potential = solvent_potential
        self.absorption_contrast = absorption_contrast
        self.voltage = voltage
        self.protein_density = protein_density
        self.molecular_weight = molecular_weight

    def calculate_box_size(self, overhang):
        return max(
            [
                self.x_limit[1] - self.x_limit[0] + 2 * overhang,
                self.y_limit[1] - self.y_limit[0] + 2 * overhang,
                self.z_limit[1] - self.z_limit[0] + 2 * overhang,
            ]
        )

    def update_limits(self):
        self.x_limit = (self.x_coordinates.min(), self.x_coordinates.max())
        self.y_limit = (self.y_coordinates.min(), self.y_coordinates.max())
        self.z_limit = (self.z_coordinates.min(), self.z_coordinates.max())

    def center_in_box(self, box_size_angstrom):
        x_size = self.x_limit[1] - self.x_limit[0]
        y_size = self.y_limit[1] - self.y_limit[0]
        z_size = self.z_limit[1] - self.z_limit[0]
        self.x_coordinates += (box_size_angstrom - x_size) / 2 - self.x_limit[0]
        self.y_coordinates += (box_size_angstrom - y_size) / 2 - self.y_limit[0]
        self.z_coordinates += (box_size_angstrom - z_size) / 2 - self.z_limit[0]
        self.update_limits()

    def select_atoms(self, x_range, y_range, z_range, voxel_size):
        x_select = np.logical_and(
            (x_range[0] * voxel_size) < self.x_coordinates,
            self.x_coordinates < (x_range[1] * voxel_size),
        )
        y_select = np.logical_and(
            (y_range[0] * voxel_size) < self.y_coordinates,
            self.y_coordinates < (y_range[1] * voxel_size),
        )
        z_select = np.logical_and(
            (z_range[0] * voxel_size) < self.z_coordinates,
            self.z_coordinates < (z_range[1] * voxel_size),
        )
        selector = np.logical_and(x_select, np.logical_and(y_select, z_select))
        # subtract start of box from the coordinates so the selection is inside the box
        return (
            self.x_coordinates[selector] - x_range[0] * voxel_size,
            self.y_coordinates[selector] - y_range[0] * voxel_size,
            self.z_coordinates[selector] - z_range[0] * voxel_size,
            self.elements[selector],
            self.b_factors[selector],
            self.occupancies[selector],
        )

    def sample_to_box(
        self,
        voxel_size=1.0,
        oversampling=1,
        box_size_angstrom=None,
        center_coordinates_in_box=True,
        overhang=20.0,
        split=1,
        gpu_id=None,
        cores=1,
    ):
        assert (
            isinstance(split, int) and split >= 1
        ), "invalid split value, only int >= 1"
        assert (
            isinstance(cores, int) and cores >= 1
        ), "invalid cores value, only int >= 1"

        # set device to run on
        if gpu_id is not None:
            assert (
                isinstance(gpu_id, int) and gpu_id >= 0
            ), "invalid gpu_id value, only int >= 0"
            device = "gpu:" + str(gpu_id)
            utils.switch_to_device(device)
        else:
            device = "cpu"

        # fix voxel size by oversampling
        if oversampling > 1:
            voxel_size /= oversampling

        if box_size_angstrom is None:
            box_size_angstrom = self.calculate_box_size(overhang)

        if center_coordinates_in_box:
            self.center_in_box(box_size_angstrom)

        # extend volume by 30 A in all directions to
        dV = voxel_size**3
        # conversion of electrostatic potential to correct units
        C = (
            4
            * np.sqrt(np.pi)
            * physics.constants["h"] ** 2
            / (physics.constants["el"] * physics.constants["me"])
            * 1e20
        )  # angstrom**2

        box_size = int(box_size_angstrom // voxel_size)
        box_size_split = int(box_size // split)
        overhang_voxels = int(overhang // voxel_size)

        if split == 1:
            # sample directly
            if "gpu" in device:
                potential, solvent = sample_iasa_gpu(
                    (
                        self.x_coordinates,
                        self.y_coordinates,
                        self.z_coordinates,
                        self.elements,
                        self.b_factors,
                        self.occupancies,
                    ),
                    (box_size,) * 3,
                    voxel_size,
                    device,
                    self.solvent_exclusion,
                )
            else:
                potential, solvent = sample_iasa_cpu(
                    (
                        self.x_coordinates,
                        self.y_coordinates,
                        self.z_coordinates,
                        self.elements,
                        self.b_factors,
                        self.occupancies,
                    ),
                    (box_size,) * 3,
                    voxel_size,
                    cores,
                    self.solvent_exclusion,
                )

        else:
            potential, solvent = np.zeros((box_size,) * 3, dtype=np.float32), np.zeros(
                (box_size,) * 3, dtype=np.float32
            )
            for i in range(split):
                for j in range(split):
                    for k in range(split):
                        x_start = box_size_split * i + int(
                            -overhang_voxels if i != 0 else 0
                        )
                        y_start = box_size_split * j + int(
                            -overhang_voxels if j != 0 else 0
                        )
                        z_start = box_size_split * k + int(
                            -overhang_voxels if k != 0 else 0
                        )
                        x_end = box_size_split * (i + 1) + int(
                            box_size % split if i == split - 1 else overhang_voxels
                        )
                        y_end = box_size_split * (j + 1) + int(
                            box_size % split if j == split - 1 else overhang_voxels
                        )
                        z_end = box_size_split * (k + 1) + int(
                            box_size % split if k == split - 1 else overhang_voxels
                        )
                        x_offset = (
                            overhang_voxels if i != 0 else 0,
                            0 if i == split - 1 else overhang_voxels,
                        )
                        y_offset = (
                            overhang_voxels if j != 0 else 0,
                            0 if j == split - 1 else overhang_voxels,
                        )
                        z_offset = (
                            overhang_voxels if k != 0 else 0,
                            0 if k == split - 1 else overhang_voxels,
                        )

                        box_sub = (x_end - x_start, y_end - y_start, z_end - z_start)

                        atoms_sub = self.select_atoms(
                            (x_start, x_end),
                            (y_start, y_end),
                            (z_start, z_end),
                            voxel_size,
                        )

                        if "gpu" in device:
                            potential_sub, solvent_sub = sample_iasa_gpu(
                                atoms_sub,
                                box_sub,
                                voxel_size,
                                device,
                                self.solvent_exclusion,
                            )
                        else:
                            potential_sub, solvent_sub = sample_iasa_cpu(
                                atoms_sub,
                                box_sub,
                                voxel_size,
                                cores,
                                self.solvent_exclusion,
                            )

                        # correct for the overhang when placing back
                        potential[
                            x_start + x_offset[0] : x_end - x_offset[1],
                            y_start + y_offset[0] : y_end - y_offset[1],
                            z_start + z_offset[0] : z_end - z_offset[1],
                        ] = potential_sub[
                            x_offset[0] : (-x_offset[1] if x_offset[1] else None),
                            y_offset[0] : (-y_offset[1] if y_offset[1] else None),
                            z_offset[0] : (-z_offset[1] if z_offset[1] else None),
                        ]
                        if solvent_sub.ndim != 1:
                            solvent[
                                x_start + x_offset[0] : x_end - x_offset[1],
                                y_start + y_offset[0] : y_end - y_offset[1],
                                z_start + z_offset[0] : z_end - z_offset[1],
                            ] = solvent_sub[
                                x_offset[0] : (-x_offset[1] if x_offset[1] else None),
                                y_offset[0] : (-y_offset[1] if y_offset[1] else None),
                                z_offset[0] : (-z_offset[1] if z_offset[1] else None),
                            ]

        # convert potential to correct units and correct for solvent exclusion
        if self.solvent_exclusion == "gaussian":
            # Correct for solvent and convert both the solvent and potential array to the correct units.
            real = (potential / dV * C) - (solvent / dV * self.solvent_potential)
        elif (
            self.solvent_exclusion == "masking"
        ):  # only if voxel size is small enough for accurate determination
            solvent_mask = (potential > 1e-5) * 1.0
            # construct solvent mask, and add gaussian decay
            if oversampling == 1:
                solvent_mask = support.reduce_resolution_real(
                    solvent_mask, voxel_size, voxel_size * 2
                )
                solvent_mask[solvent_mask < 0.001] = 0
            # subtract solvent from the protein electrostatic potential
            real = (potential / dV * C) - (solvent_mask * self.solvent_potential)
        else:
            real = potential / dV * C

        # determine absorption contrast if set
        if self.absorption_contrast:
            # voltage by default 300 keV
            molecule_absorption = physics.potential_amplitude(
                self.protein_density, self.molecular_weight, self.voltage
            )
            solvent_absorption = physics.potential_amplitude(
                physics.AMORPHOUS_ICE_DENSITY, physics.WATER_MW, self.voltage
            ) * (self.solvent_potential / physics.V_WATER)
            print("Calculating absorption contrast")
            print(f"Molecule absorption = {molecule_absorption:.3f}")
            print(f"Solvent absorption = {solvent_absorption:.3f}")

            if self.solvent_exclusion == "masking":
                imaginary = solvent_mask * (molecule_absorption - solvent_absorption)
            elif self.solvent_exclusion == "gaussian":
                imaginary = solvent / dV * (molecule_absorption - solvent_absorption)
            else:
                print(
                    "ERROR: Absorption contrast cannot be generated if solvent exclusion is not set to either gaussian "
                    "or masking."
                )
                sys.exit(0)

            electrostatic_potential = real + 1j * imaginary

        else:
            electrostatic_potential = real

        if oversampling > 1:
            print("Rescaling after oversampling")
            electrostatic_potential = ndimage.zoom(
                support.reduce_resolution_real(
                    electrostatic_potential, voxel_size, voxel_size * 2 * oversampling
                ),
                1 / oversampling,
                order=3,
            )

        return electrostatic_potential


def iasa_integration(
    filepath,
    voxel_size=1.0,
    oversampling=1,
    solvent_exclusion=None,
    V_sol=physics.V_WATER,
    absorption_contrast=False,
    voltage=300e3,
    density=physics.PROTEIN_DENSITY,
    molecular_weight=physics.PROTEIN_MW,
    structure_tuple=None,
):
    """
    Calculates interaction potential map to 1 A volume as described initially by Rullgard et al. (2011) in TEM
    simulator, but adapted from matlab InSilicoTEM from Vulovic et al. (2013). This function applies averaging of
    the potential over the voxels to obtain precise results without oversampling.

    @param filepath: full filepath to pdb file
    @type  filepath: L{string}
    @param voxel_size: size of voxel in output map, default 1 A
    @type  voxel_size: L{float}
    @param oversampling: number of times to oversample final voxel size
    @type  oversampling: L{int}
    @param solvent_exclusion: flag to execute solvent exclusion using gaussian spheres. Solvent exclusion can be set
    with a string, either 'gaussian' or 'masking'. Default is None.
    @type  solvent_exclusion: L{str}
    @param V_sol: average solvent background potential (V/A^3)
    @type  V_sol: L{float}
    @param absorption_contrast: flag to generate absorption factor for imaginary part of potential
    @type  absorption_contrast: L{bool}
    @param voltage: electron beam voltage, absorption factor depends on voltage, default 300e3
    @type  voltage: L{float}
    @param density: average density of molecule that is generated, default 1.35 (protein)
    @type  density: L{float}
    @param molecular_weight: average molecular weight of the molecule that is generated, default protein MW
    @type  molecular_weight: L{float}
    @param structure_tuple: structure information as a tuple (x_coordinates, y_coordinates, z_coordinates, elements,
    b_factors, occupancies), if provided this overrides file reading
    @type  structure_tuple: L{tuple} - (L{list},) * 6 with types (float, float, float, str, float, float)

    @return: A volume with interaction potentials, either tuple of (real, imag) or single real, both real and imag
    are 3d arrays.
    @rtype: L{tuple} -> (L{np.ndarray},) * 2 or L{np.ndarray}

    @author: Marten Chaillet
    """
    from scipy.special import erf

    assert (type(oversampling) is int) and (oversampling >= 1), print(
        "oversampling parameter is not an integer"
    )
    if oversampling > 1:
        voxel_size /= oversampling

    extra_space = 30  # extend volume by 30 A in all directions

    print(f" - Calculating IASA potential from {filepath}")

    if structure_tuple is None:
        (
            x_coordinates,
            y_coordinates,
            z_coordinates,
            elements,
            b_factors,
            occupancies,
        ) = pdbs.read_structure(filepath)
    else:
        (
            x_coordinates,
            y_coordinates,
            z_coordinates,
            elements,
            b_factors,
            occupancies,
        ) = structure_tuple

    x_max = np.max(x_coordinates - np.min(x_coordinates))
    y_max = np.max(y_coordinates - np.min(y_coordinates))
    z_max = np.max(z_coordinates - np.min(z_coordinates))
    dimensions = [x_max, y_max, z_max]
    largest_dimension = max(dimensions)
    difference = [largest_dimension - a for a in dimensions]

    x_coordinates = (
        x_coordinates - np.min(x_coordinates) + extra_space + difference[0] / 2
    )
    y_coordinates = (
        y_coordinates - np.min(y_coordinates) + extra_space + difference[1] / 2
    )
    z_coordinates = (
        z_coordinates - np.min(z_coordinates) + extra_space + difference[2] / 2
    )
    # Define the volume of the protein
    sz = (int((largest_dimension + 2 * extra_space) / voxel_size),) * 3

    potential = np.zeros(sz)
    if solvent_exclusion:
        solvent = np.zeros(sz)

    print(f"Number of atoms to go over is {len(x_coordinates)}")

    for i in range(len(elements)):
        if np.mod(i, 5000) == 0:
            print(f"Calculating atom {i}.")

        atom = elements[i].upper()
        b_factor = b_factors[i]
        occupancy = occupancies[i]

        sf = np.array(physics.scattering_factors[atom]["g"])
        a = sf[0:5]
        b = sf[5:10]

        # b += (b_factor) # units in A

        if atom in list(physics.volume_displaced):
            r_0 = np.cbrt(physics.volume_displaced[atom] / (np.pi ** (3 / 2)))
        else:  # If not H,C,O,N we assume the same volume displacement as for carbon
            r_0 = np.cbrt(physics.volume_displaced["C"] / (np.pi ** (3 / 2)))

        r2 = 0  # this was 15 / (1 / r_0 ** 2) before but this gives very high values for carbon
        for j in range(5):
            # Find the max radius over all gaussians (assuming symmetrical potential to 4.5 sigma truncation
            # (corresponds to 10).
            r2 = np.maximum(r2, 15 / (4 * np.pi**2 / b[j]))
        # Radius of gaussian sphere
        r = np.sqrt(r2 / 3)

        rc = [x_coordinates[i], y_coordinates[i], z_coordinates[i]]  # atom center
        ind_min = [
            int((c - r) // voxel_size) for c in rc
        ]  # Smallest index to contain relevant potential x,y,z
        ind_max = [
            int((c + r) // voxel_size) for c in rc
        ]  # Largest index to contain relevant potential x,y,z
        # Explicit real space coordinates for the max and min boundary of each voxel
        x_min_bound = np.arange(ind_min[0], ind_max[0] + 1, 1) * voxel_size - rc[0]
        x_max_bound = np.arange(ind_min[0] + 1, ind_max[0] + 2, 1) * voxel_size - rc[0]
        y_min_bound = np.arange(ind_min[1], ind_max[1] + 1, 1) * voxel_size - rc[1]
        y_max_bound = np.arange(ind_min[1] + 1, ind_max[1] + 2, 1) * voxel_size - rc[1]
        z_min_bound = np.arange(ind_min[2], ind_max[2] + 1, 1) * voxel_size - rc[2]
        z_max_bound = np.arange(ind_min[2] + 1, ind_max[2] + 2, 1) * voxel_size - rc[2]

        atom_potential = 0

        for j in range(5):
            sqrt_b = np.sqrt(b[j])  # calculate only once
            # Difference of error function == integrate over Gaussian
            int_x = (
                sqrt_b
                / (4 * np.sqrt(np.pi))
                * (
                    erf(x_max_bound * 2 * np.pi / sqrt_b)
                    - erf(x_min_bound * 2 * np.pi / sqrt_b)
                )
            )
            x_matrix = np.tile(
                int_x[:, np.newaxis, np.newaxis],
                [1, ind_max[1] - ind_min[1] + 1, ind_max[2] - ind_min[2] + 1],
            )
            int_y = (
                sqrt_b
                / (4 * np.sqrt(np.pi))
                * (
                    erf(y_max_bound * 2 * np.pi / sqrt_b)
                    - erf(y_min_bound * 2 * np.pi / sqrt_b)
                )
            )
            y_matrix = np.tile(
                int_y[np.newaxis, :, np.newaxis],
                [ind_max[0] - ind_min[0] + 1, 1, ind_max[2] - ind_min[2] + 1],
            )
            int_z = (
                sqrt_b
                / (4 * np.sqrt(np.pi))
                * (
                    erf(z_max_bound * 2 * np.pi / sqrt_b)
                    - erf(z_min_bound * 2 * np.pi / sqrt_b)
                )
            )
            z_matrix = np.tile(
                int_z[np.newaxis, np.newaxis, :],
                [ind_max[0] - ind_min[0] + 1, ind_max[1] - ind_min[1] + 1, 1],
            )

            atom_potential += a[j] / b[j] ** (3 / 2) * x_matrix * y_matrix * z_matrix

        # scatter_add instead of +=
        potential[
            ind_min[0] : ind_max[0] + 1,
            ind_min[1] : ind_max[1] + 1,
            ind_min[2] : ind_max[2] + 1,
        ] += atom_potential

        if solvent_exclusion == "gaussian":
            # excluded solvent potential
            int_x = (
                np.sqrt(np.pi)
                * r_0
                / 2
                * (erf(x_max_bound / r_0) - erf(x_min_bound / r_0))
            )
            x_matrix = np.tile(
                int_x[:, np.newaxis, np.newaxis],
                [1, ind_max[1] - ind_min[1] + 1, ind_max[2] - ind_min[2] + 1],
            )
            int_y = (
                np.sqrt(np.pi)
                * r_0
                / 2
                * (erf(y_max_bound / r_0) - erf(y_min_bound / r_0))
            )
            y_matrix = np.tile(
                int_y[np.newaxis, :, np.newaxis],
                [ind_max[0] - ind_min[0] + 1, 1, ind_max[2] - ind_min[2] + 1],
            )
            int_z = (
                np.sqrt(np.pi)
                * r_0
                / 2
                * (erf(z_max_bound / r_0) - erf(z_min_bound / r_0))
            )
            z_matrix = np.tile(
                int_z[np.newaxis, np.newaxis, :],
                [ind_max[0] - ind_min[0] + 1, ind_max[1] - ind_min[1] + 1, 1],
            )

            solvent[
                ind_min[0] : ind_max[0] + 1,
                ind_min[1] : ind_max[1] + 1,
                ind_min[2] : ind_max[2] + 1,
            ] += (
                x_matrix * y_matrix * z_matrix
            )

    # Voxel volume
    dV = voxel_size**3
    # Convert to correct units
    C = (
        4
        * np.sqrt(np.pi)
        * physics.constants["h"] ** 2
        / (physics.constants["el"] * physics.constants["me"])
        * 1e20
    )  # angstrom**2

    if solvent_exclusion == "gaussian":
        # Correct for solvent and convert both the solvent and potential array to the correct units.
        real = (potential / dV * C) - (solvent / dV * V_sol)
    elif (
        solvent_exclusion == "masking"
    ):  # only if voxel size is small enough for accurate determination of mask
        solvent_mask = (potential > 1e-5) * 1.0
        # construct solvent mask
        # gaussian decay of mask
        if oversampling == 1:
            solvent_mask = support.reduce_resolution_fourier(
                solvent_mask, voxel_size, voxel_size * 2
            )
            solvent_mask[solvent_mask < 0.001] = 0
        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # slice = int(solvent_mask.shape[2] // 2)
        # ax1.imshow(potential[:, :, slice])
        # ax2.imshow(solvent_mask[:, :, slice])
        # show()
        real = (potential / dV * C) - (solvent_mask * V_sol)
    else:
        real = potential / dV * C

    if absorption_contrast:
        # voltage by default 300 keV
        molecule_absorption = physics.potential_amplitude(
            density, molecular_weight, voltage
        )
        solvent_absorption = physics.potential_amplitude(
            physics.AMORPHOUS_ICE_DENSITY, physics.WATER_MW, voltage
        ) * (V_sol / physics.V_WATER)
        print(f"molecule absorption = {molecule_absorption:.3f}")
        print(f"solvent absorption = {solvent_absorption:.3f}")

        if solvent_exclusion == "masking":
            imaginary = solvent_mask * (molecule_absorption - solvent_absorption)
        elif solvent_exclusion == "gaussian":
            imaginary = solvent / dV * (molecule_absorption - solvent_absorption)
        else:
            print(
                "ERROR: Absorption contrast cannot be generated if the solvent masking or solvent exclusion method "
                "are not used."
            )
            sys.exit(0)

        real = support.reduce_resolution_fourier(
            real, voxel_size, voxel_size * 2 * oversampling
        )
        real = ndimage.zoom(real, 1 / oversampling, order=3)
        imaginary = support.reduce_resolution_fourier(
            imaginary, voxel_size, voxel_size * 2 * oversampling
        )
        imaginary = ndimage.zoom(imaginary, 1 / oversampling, order=3)
        return real + 1j * imaginary
    else:
        real = support.reduce_resolution_fourier(
            real, voxel_size, voxel_size * 2 * oversampling
        )
        return ndimage.zoom(real, 1 / oversampling, order=3)


def wrapper(
    filepath,
    output_folder,
    voxel_size,
    skip_structure_edit=False,
    oversampling=1,
    binning=1,
    solvent_exclusion=None,
    solvent_potential=physics.V_WATER,
    absorption_contrast=False,
    voltage=300e3,
    solvent_factor=1.0,
    cores=1,
    gpu_id=None,
):
    """
    Execution of generating an electrostatic potential (and absorption potential) from a pdb/cif file. Process
    includes preprocessing with chimera to add hydrogens and symmetry, then passing to IASA_intergration method to
    correctly sample the electrostatic potential to a 3d array. Two options can be provided for solvent correction.

    @param filepath: full path to pdb or cif filed
    @type  filepath: L{str}
    @param output_folder: folder to write all output to
    @type  output_folder: L{str}
    @param voxel_size: voxel size in A to sample the interaction potential to.
    @type  voxel_size: L{float}
    @param oversampling: number of times to oversample the interaction potential for better accuracy, multiple of 1
    @type  oversampling: L{int}
    @param binning: number of times to bin the volume after sampling, this file will be saved separately
    @type  binning: L{int}
    @param exclude_solvent: flag to exclude solvent with a Gaussian sphere
    @type  exclude_solvent: L{bool}
    @param solvent_masking: flag to excluded solvent by masking (thresholding method)
    @type  solvent_masking: L{bool}
    @param solvent_potential: background solvent potential, default 4.5301
    @type  solvent_potential: L{float}
    @param absorption_contrast: flag for generating absorption potential
    @type  absorption_contrast: L{bool}
    @param voltage: electron beam voltage in eV, parameter for absorption contrast, default 300e3
    @type  voltage: L{float}
    @param solvent_factor: solvent factor to increase background potential
    @type  solvent_factor: L{float}

    @return: - (files are written to output_folder)
    @rtype:  Nonee

    @author: Marten Chaillet
    """
    # Id does not makes sense to apply absorption contrast if solvent exclusion is not turned on
    if absorption_contrast:
        assert solvent_exclusion is not None, print(
            "absorption contrast can only be applied if solvent exclusion is " "used."
        )

    _, filename = os.path.split(filepath)
    pdb_id, _ = os.path.splitext(filename)

    # Call external programs for structure preparation and PB-solver
    if not skip_structure_edit:
        try:
            filepath = pdbs.call_chimera(filepath, output_folder)  # output structure
            # name is dependent on
            # modification by chimera
        except utils.StructureModificationError as e:
            print("Modification went wrong, I will continue with the base pdb.")

    assert filepath != 0, "something went wrong with chimera"

    # Calculate atom and bond potential, and store them
    # 4 times oversampling of IASA yields accurate potentials
    # Could be {structure}.{extension}, but currently chimera is only able to produce .pdb files, so the extended
    # structure file created by call chimera has a .pdb extension.
    ep = ElectrostaticPotential(
        filepath,
        solvent_exclusion=solvent_exclusion,
        solvent_potential=solvent_potential * solvent_factor,
        absorption_contrast=absorption_contrast,
        voltage=voltage,
    )
    v_atom = ep.sample_to_box(
        voxel_size=voxel_size,
        oversampling=oversampling,
        center_coordinates_in_box=True,
        overhang=30,
        split=1,
        gpu_id=gpu_id,
        cores=cores,
    )

    # Absorption contrast map generated here will look blocky when generated at 2.5A and above!
    if np.iscomplexobj(v_atom):
        output_name = f"{pdb_id}_{voxel_size:.2f}A_solvent-{solvent_potential*solvent_factor:.3f}V"
        print(f"writing real and imaginary part with name {output_name}")
        support.write_mrc(
            os.path.join(output_folder, f"{output_name}_real.mrc"),
            v_atom.real,
            voxel_size,
        )
        support.write_mrc(
            os.path.join(output_folder, f"{output_name}_imag_{voltage*1E-3:.0f}V.mrc"),
            v_atom.imag,
            voxel_size,
        )
    else:
        if solvent_exclusion:
            output_name = f"{pdb_id}_{voxel_size:.2f}A_solvent-{solvent_potential*solvent_factor:.3f}V"
        else:
            output_name = f"{pdb_id}_{voxel_size:.2f}A"
        print(f"writing real part with name {output_name}")
        support.write_mrc(
            os.path.join(output_folder, f"{output_name}_real.mrc"), v_atom, voxel_size
        )

    if binning > 1:

        print(" - Binning volume")
        downsampled = ndimage.zoom(
            support.reduce_resolution_real(
                v_atom, voxel_size, voxel_size * 2 * binning
            ),
            1 / binning,
            order=3,
        )

        if np.iscomplexobj(v_atom):

            output_name = f"{pdb_id}_{voxel_size*binning:.2f}A_solvent-{solvent_potential*solvent_factor:.3f}V"
            print(f"writing real and imaginary part with name {output_name}")
            support.write_mrc(
                os.path.join(output_folder, f"{output_name}_real.mrc"),
                downsampled.real,
                voxel_size * binning,
            )
            support.write_mrc(
                os.path.join(
                    output_folder, f"{output_name}_imag_{voltage * 1E-3:.0f}V.mrc"
                ),
                downsampled.imag,
                voxel_size * binning,
            )

        else:

            if solvent_exclusion:
                output_name = f"{pdb_id}_{voxel_size*binning:.2f}A_solvent-{solvent_potential*solvent_factor:.3f}V"
            else:
                output_name = f"{pdb_id}_{voxel_size*binning:.2f}A"
            print(f"writing real part with name {output_name}")
            support.write_mrc(
                os.path.join(output_folder, f"{output_name}_real.mrc"),
                downsampled,
                voxel_size * binning,
            )
    return
