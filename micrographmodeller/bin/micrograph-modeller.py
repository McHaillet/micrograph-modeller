#!/usr/bin/env python

import tracemalloc
import configparser
import os
import sys
import logging
import datetime
import numpy as np
import random
import micrographmodeller as mm
from ast import literal_eval
from importlib import resources as importlib_resources


class ConfigLogger(object):
    """
    Facilitates writing the conf file to a .log file in the output_folder for reference of settings.
    """
    def __init__(self, log):
        self.__log = log

    def __call__(self, config):
        self.__log.info("Config:")
        config.write(self)

    def write(self, data):
        # stripping the data makes the output nicer and avoids empty lines
        line = data.strip()
        self.__log.info(line)


def draw_range(range, datatype, name):
    """
    Input parsing from config file. This parses possible ranges of values and randomly samples in the range. In case
    the input is just a single value that will be used instead and no random selection will be done. This allows users
    to dynamically specify a range or single value depending on the needs.

    @param range: list of two values to select in between
    @type  range: L{list} - [L{float},] * 2
    @param datatype: desired type for the parameter, either int or float
    @type  datatype: L{str}
    @param name: name of the simulation parameter
    @type  name: L{str}

    @return: a single value that was selected from the range, or single parsed value
    @rtype:  L{int} or L{float}
    """
    if type(range) == list and len(range) == 2:
        np.random.seed(seed)
        random.seed(seed)
        if datatype == int:
            return np.random.randint(range[0], range[1])
        elif datatype == float:
            return np.random.uniform(range[0], range[1])
    elif type(range) == list and len(range) == 1:
        if datatype == int:
            return int(range[0])
        elif datatype == float:
            return float(range[0])
    elif type(range) == float or type(range) == int:
        if datatype == int:
            return int(range)
        elif datatype == float:
            return float(range)
    else:
        print(f'invalid data range or input type for parameter {name}')
        sys.exit(0)


if __name__ == '__main__':
    # ------------------------------Import functions used in main-------------------------------------------------------
    # loadstar is for reading .meta files containing data collection parameters (tilt angles, etc.).
    # literal_eval is used for passing arguments from the config file.
    # Use tracemalloc to record the peak memory usage of the program
    tracemalloc.start()

    # --------------------------------------Read config-----------------------------------------------------------------
    config = configparser.ConfigParser()
    try:
        if len(sys.argv) > 1:
            config_given = sys.argv[1]
            if config_given and os.path.exists(config_given):
                print(f'\nLoading a given configuration file: {config_given}')
                config.read_file(open(config_given))
        else:
            print(f'\nLoading default configuration file: pytom/simulation/simulation.conf')
            config.read_file(open('simulation.conf'))
    except Exception as e:
        print(e)
        raise Exception('Could not open config file.')

    print('Configuration sections:', config.sections())

    # ----------------------------------------Set simulation parameters-------------------------------------------------
    try:
        output_folder           = config['General']['OutputFolder']
        simulator_mode          = config['General']['Mode']
        device                  = config['General']['Device']
        nodes                   = config['General'].getint('Nodes')
        model_ID                = config['General'].getint('ModelID')
        seed                    = config['General'].getint('Seed')
        pixel_size              = config['General'].getfloat('PixelSize') * 1E-10 # pixel_size in nm
        oversampling            = config['General'].getint('Oversampling')  # oversampling is used for correcting
        # poisson statistics and camera DQE and MTF functions
        solvent_potential       = config['General'].getfloat('SolventConstant')
        absorption_contrast     = config['General'].getboolean('AbsorptionContrast')
        voltage                 = config['General'].getfloat('Voltage') * 1E3  # voltage in keV
        # voltage and pixelsize are needed for model generation and projection, thus general parameters

        # ensure simulator mode and device are valid options
        if (simulator_mode in ['TiltSeries', 'FrameSeries']) or (device in ['CPU', 'GPU']):
            print(f'Generating model {model_ID} on {device} in folder {output_folder}')
        else:
            print('Invalid entry for simulator mode or device in config.')
            sys.exit(0)
    except Exception as e:
        print(e)
        raise Exception('Missing general parameters in config file.')

    if 'GenerateModel' in config.sections():
        try:
            # We assume the particle models are in the desired voxel spacing for the pixel size of the simulation!
            particle_folder     = config['GenerateModel']['ParticleFolder']
            listpdbs            = literal_eval(config['GenerateModel']['Models'])
            listmembranes       = literal_eval(config['GenerateModel']['MembraneModels'])
            size                = config['GenerateModel'].getint('Size')
            placement_size      = config['GenerateModel'].getint('PlacementSize')
            # parse range of ice thickness, provided in nm
            thickness           = draw_range(literal_eval(config['GenerateModel']['Thickness']), float, 'Thickness') * 1E-9
            thickness_voxels    = int(thickness / pixel_size) # calculate thickness in number of voxels!
            # make even number to solve tomogram reconstruction mismatch bug
            thickness_voxels -= (thickness_voxels % 4)
            # gold markers
            number_of_markers   = draw_range(literal_eval(config['GenerateModel']['NumberOfMarkers']), int,
                                           'NumberOfMarkers')
            # parse range of number of particles
            number_of_particles = draw_range(literal_eval(config['GenerateModel']['NumberOfParticles']), int,
                                             'NumberOfParticles')
            number_of_membranes = draw_range(literal_eval(config['GenerateModel']['NumberOfMembranes']), int,
                                             'NumberOfMembranes')
            sigma_motion_blur   = config['GenerateModel'].getfloat('SigmaMotionBlur')  # in A units

            # TODO add parameter for meta mode of random variation or stick exactly to input values
        except Exception as e:
            print(e)
            raise Exception('Missing generate model parameters in config file.')

    if 'Microscope' in config.sections():
        try:
            camera                  = config['Microscope']['Camera']
            try:
                camera_folder       = config['Microscope']['CameraFolder']
            except Exception as e:
                with importlib_resources.path(mm, 'detectors') as path:
                    camera_folder   = str(path)
            # beam damage SNR
            beam_damage_snr         = draw_range(literal_eval(config['Microscope']['BeamDamageSNR']), float,
                                                 'BeamDamageSNR')
            defocus                 = draw_range(literal_eval(config['Microscope']['Defocus']), float, 'Defocus') * 1E-6
            electron_dose           = draw_range(literal_eval(config['Microscope']['ElectronDose']), float, 'ElectronDose')
            spherical_aberration    = config['Microscope'].getfloat('SphericalAberration') * 1E-3
            chromatic_aberration    = config['Microscope'].getfloat('ChromaticAberration') * 1E-3
            energy_spread           = config['Microscope'].getfloat('EnergySpread')
            illumination_aperture   = config['Microscope'].getfloat('IlluminationAperture') * 1E-3
            objective_diameter      = config['Microscope'].getfloat('ObjectiveDiameter') * 1E-6
            focus_length            = config['Microscope'].getfloat('FocalDistance') * 1E-3
            astigmatism             = config['Microscope'].getfloat('Astigmatism') * 1E-9
            astigmatism_angle       = draw_range(literal_eval(config['Microscope']['AstigmatismAngle']), float,
                                                 'AstigmatismAngle')
        except Exception as e:
            print(e)
            raise Exception('Missing microscope parameters in config file.')

    if simulator_mode in config.sections():
        try:
            # first read common parameters between tilt and frame series
            image_size              = config[simulator_mode].getint('ImageSize')
            msdz                    = config[simulator_mode].getfloat('MultisliceStep') * 1E-9
            # random translations between frames/tilts in A
            translation_shift       = draw_range(literal_eval(config[simulator_mode]['TranslationalShift']), float,
                                                 'TranslationalShift')
            # mode specific parameters
            if simulator_mode == 'TiltSeries':  # make increment scheme  ?  but allow for more complex variation?
                metadata            = mm.support.loadstar(config['TiltSeries']['MetaFile'],
                                                       dtype=mm.support.DATATYPE_METAFILE)
                angles              = metadata['TiltAngle']  # in degrees
            elif simulator_mode == 'FrameSeries':
                number_of_frames    = config['FrameSeries'].getint('NumberOfFrames')
        except Exception as e:
            print(e)
            raise Exception(f'Missing {simulator_mode} parameters in config file.')

    if 'ScaleProjections' in config.sections():
        try:
            example_folder      = config['ScaleProjections']['ExampleFolder']
            example_pixel_size  = config['ScaleProjections'].getfloat('ExamplePixelSize')
            # If experimental and simulated projections have different size, we need to crop. This should be done with
            # care if the option oversampling is set for reconstructions, because in that case the ground truth data needs
            # to be binned and cropped as well. Uneven size of the volume means the ground truth data will be shifted
            # by half a pixel compared to the reconstruction. This options makes sure that this not happen.
            make_even_factor    = config['ScaleProjections'].getint('EvenSizeFactor')
        except Exception as e:
            print(e)
            raise Exception('Missing experimental projection scaling parameters.')

    if 'TomogramReconstruction' in config.sections():
        try:
            reconstruction_bin      = config['TomogramReconstruction'].getint('Binning')
            use_scaled_projections  = config['TomogramReconstruction'].getboolean('UseScaledProjections')
            align                   = config['TomogramReconstruction'].getboolean('Align')
        except Exception as e:
            print(e)
            raise Exception('Missing tomogram reconstruction parameters in config file.')

    # --------------------------------------Create directories and logger-----------------------------------------------
    save_path = os.path.join(output_folder, f'model_{model_ID}')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    logging.basicConfig(filename='{}/simulator-{date:%Y-%m-%d_%H:%M:%S}.log'.format(save_path,
                                                                date=datetime.datetime.now()), level=logging.INFO)
    config_logger = ConfigLogger(logging)
    config_logger(config)

    logging.info('Values of parameters that we randomly vary per simulation (only for generate model and generate projections):')
    if 'GenerateModel' in config.sections():
        logging.info(f'model thickness = {thickness_voxels*pixel_size*1E9:.2f}nm (adjusted to be an even number of voxels)')
        logging.info(f'# of particles = {number_of_particles}')
        logging.info(f'# of markers = {number_of_markers}')
        logging.info(f'# of membranes = {number_of_membranes}')
    if 'Microscope' in config.sections():
        logging.info(f'defocus = {defocus*1E6:.2f}um')
        logging.info(f'electron dose = {electron_dose} e-/A^2')

    # ----------------------------------------Execute simulation--------------------------------------------------------
    if 'GenerateModel' in config.sections():
        # set seed for random number generation
        np.random.seed(seed)
        random.seed(seed)

        print('\n- Generating grand model')
        mm.micrographmodeller.generate_model(particle_folder, save_path, listpdbs, listmembranes,
                       pixel_size           =pixel_size * 1E10,
                       size                 =size,
                       thickness            =thickness_voxels,
                       placement_size       =placement_size,
                       solvent_potential    =solvent_potential,
                       number_of_particles  =number_of_particles,
                       number_of_markers    =number_of_markers,
                       absorption_contrast  =absorption_contrast,
                       voltage              =voltage,
                       number_of_membranes  =number_of_membranes,
                       sigma_motion_blur    =sigma_motion_blur)

    if simulator_mode in config.sections() and simulator_mode == 'TiltSeries':
        # set seed for random number generation
        np.random.seed(seed)
        random.seed(seed)
        # Grab the ice thickness from the initial model in case program is only executed for projections
        print('\n- Generating projections')
        if device == 'CPU':
            mm.micrographmodeller.generate_tilt_series_cpu(save_path, angles,
                                      nodes                 =nodes,
                                      image_size            =image_size,
                                      rotation_box_height   =None,  # will automatically calculate fitting size if None
                                      pixel_size            =pixel_size,
                                      oversampling          =oversampling,
                                      dose                  =electron_dose,
                                      voltage               =voltage,
                                      spherical_aberration  =spherical_aberration,
                                      chromatic_aberration  =chromatic_aberration,
                                      energy_spread         =energy_spread,
                                      illumination_aperture =illumination_aperture,
                                      objective_diameter    =objective_diameter,
                                      focus_length          =focus_length,
                                      astigmatism           =astigmatism,
                                      astigmatism_angle     =astigmatism_angle,
                                      msdz                  =msdz,
                                      defocus               =defocus,
                                      sigma_shift           =translation_shift,
                                      camera_type           =camera,
                                      camera_folder         =camera_folder,
                                      solvent_potential     =solvent_potential,
                                      absorption_contrast   =absorption_contrast,
                                      beam_damage_snr       =beam_damage_snr)
        elif device == 'GPU':
            print('This option needs to be implemented.')
            sys.exit(0)
        else:
            print('Invalid device type.')
            sys.exit(0)
    elif simulator_mode in config.sections() and simulator_mode == 'FrameSeries':
        np.random.seed(seed)
        random.seed(seed)
        print('\n- Generate frame series projections')
        if device == 'CPU':
            mm.micrographmodeller.generate_frame_series_cpu(save_path,
                                      n_frames              =number_of_frames,
                                      nodes                 =nodes,
                                      image_size            =image_size,
                                      pixel_size            =pixel_size,
                                      oversampling          =oversampling,
                                      dose                  =electron_dose,
                                      voltage               =voltage,
                                      spherical_aberration  =spherical_aberration,
                                      chromatic_aberration  =chromatic_aberration,
                                      energy_spread         =energy_spread,
                                      illumination_aperture =illumination_aperture,
                                      objective_diameter    =objective_diameter,
                                      focus_length          =focus_length,
                                      astigmatism           =astigmatism,
                                      astigmatism_angle     =astigmatism_angle,
                                      msdz                  =msdz,
                                      defocus               =defocus,
                                      mean_shift            =translation_shift,
                                      camera_type           =camera,
                                      camera_folder         =camera_folder,
                                      solvent_potential     =solvent_potential,
                                      absorption_contrast   =absorption_contrast,
                                      beam_damage_snr       =beam_damage_snr)
        elif device == 'GPU':
            print('This option needs to be implemented.')
            sys.exit(0)
        else:
            print('Invalid device type.')
            sys.exit(0)

    if 'ScaleProjections' in config.sections():
        # set seed for random number generation
        np.random.seed(seed)
        random.seed(seed)
        print('\n- Scaling projections with experimental data')
        mm.micrographmodeller.scale_projections(save_path, pixel_size * 1E10, example_folder,
                                            example_pixel_size, oversampling, nodes, make_even_factor)

    if 'TomogramReconstruction' in config.sections():
        print('\n- Reconstructing tomogram')
        mm.micrographmodeller.reconstruct_tomogram(save_path,
                             binning=reconstruction_bin,
                             use_scaled_projections=use_scaled_projections,
                             align_projections=align)

    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    tracemalloc.stop()