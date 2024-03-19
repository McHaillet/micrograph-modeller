import unittest
import numpy as np
import os
import micrographmodeller as mm
from micrographmodeller import potential, simulator, support
from importlib import resources as importlib_resources


class TestMicrographModeller(unittest.TestCase):
    def setUp(self):
        """Initialize simulation parameters"""

        self.param_pot = {
            "pdb": "test_data/3j9m.cif",
            "voxel_size": 5,
            "oversampling": 2,
            "solvent_exclusion": "gaussian",
            "absorption_contrast": True,
            "voltage": 300e3,
        }

        ep = potential.ElectrostaticPotential(
            self.param_pot["pdb"],
            solvent_exclusion=self.param_pot["solvent_exclusion"],
            absorption_contrast=self.param_pot["absorption_contrast"],
            voltage=self.param_pot["voltage"],
        )
        self.potential = ep.sample_to_box(
            voxel_size=self.param_pot["voxel_size"],
            oversampling=self.param_pot["oversampling"],
            center_coordinates_in_box=True,
            overhang=30,
            split=1,
            cores=4,
        )

        if self.potential.shape[0] % 2:
            self.potential = np.pad(
                self.potential, pad_width=(0, 1), mode="constant", constant_values=0
            )

        # create temporary dir for storing simulation data
        if not os.path.exists("temp_simulation"):
            os.mkdir("temp_simulation")

        # camera folder
        with importlib_resources.path(mm, "detectors") as path:
            camera_folder = str(path)

        # specific defocus and msdz, but otherwise default parameters for ctf function
        self.param_sim = {
            "save_path": "./temp_simulation",
            "angles": list(range(-60, 60 + 3, 3)),
            "nodes": 1,  # todo change to multiple if possible ??
            "pixel_size": 5e-10,
            "oversampling": 2,
            "dose": 80,
            "voltage": 300e3,
            "defocus": 3e-6,
            "msdz": 5e-9,
            "camera_type": "K2SUMMIT",
            "camera_folder": camera_folder,
        }

        self.param_rec = {"save_path": "./temp_simulation", "reconstruction_bin": 1}

    def tearDown(self):
        """Remove all the files gathered during simulation"""
        directory = self.param_sim["save_path"]
        self.remove_file(os.path.join(directory, "projections.mrc"))
        self.remove_file(os.path.join(directory, "noisefree_projections.mrc"))
        self.remove_file(os.path.join(directory, "simulation.meta"))
        self.remove_dir(directory)

    def remove_dir(self, foldername):
        """Assert folder exists, then remove its content and itself"""
        foldercheck = os.path.exists(foldername)
        if not foldercheck:
            print(foldername + " does not exist")
        self.assertTrue(foldercheck, msg="folder " + foldername + " does not exist")
        if foldercheck:
            os.rmdir(foldername)

    def remove_file(self, filename):
        """Assert that file exists, then remove it"""
        filecheck = os.path.exists(filename)
        if not filecheck:
            print(filename + " does not exist")
        self.assertTrue(filecheck, msg="file " + filename + " does not exist")
        if filecheck:
            os.remove(filename)

    def simulate_tomogram(self, c=""):
        """Run the simulation, output here will be written to some temp storage"""

        if not os.path.exists(self.param_sim["save_path"] + c):
            os.mkdir(self.param_sim["save_path"] + c)

        simulator.generate_tilt_series_cpu(
            self.param_sim["save_path"] + c,
            self.param_sim["angles"],
            nodes=self.param_sim["nodes"],
            pixel_size=self.param_sim["pixel_size"],
            oversampling=self.param_sim["oversampling"],
            dose=self.param_sim["dose"],
            voltage=self.param_sim["voltage"],
            defocus=self.param_sim["defocus"],
            msdz=self.param_sim["msdz"],
            camera_type=self.param_sim["camera_type"],
            camera_folder=self.param_sim["camera_folder"],
            grandcell=self.potential.copy(),
        )

        # reconstruct the tomogram with alignment
        metadata = support.loadstar(
            os.path.join(self.param_rec["save_path"] + c, "simulation.meta"),
            dtype=support.DATATYPE_METAFILE,
        )
        alignment = [
            (
                m["TiltAngle"],
                m["InPlaneRotation"],
                m["TranslationX"],
                m["TranslationY"],
                m["Magnification"],
            )
            for m in metadata
        ]
        projections, _ = support.read_mrc(
            os.path.join(self.param_rec["save_path"] + c, "projections.mrc")
        )
        return simulator.weighted_back_projection(
            projections, alignment, self.potential.shape, (0, 0, 0), 1
        )

    def test(self):
        """Run two simulations and test their correlation. Both will have a different realization of noise and will
        slightly differ."""

        # generate two different realization of tomogram noise
        spacing = self.param_sim["pixel_size"] * 1e10
        tomo_1 = self.simulate_tomogram()
        tomo_2 = self.simulate_tomogram()
        # support.write_mrc('./subtomo1.mrc', tomo_1, 5)
        # support.write_mrc('./subtomo2.mrc', tomo_2, 5)

        tomo_1 = support.reduce_resolution_fourier(
            tomo_1, spacing, 2 * spacing * 8
        )
        tomo_2 = support.reduce_resolution_fourier(
            tomo_2, spacing, 2 * spacing * 8
        )

        # mask for correlation
        r = int(tomo_1.shape[0] / 2 * 0.8)
        mask = support.create_sphere(
            tomo_1.shape, radius=r, sigma=r / 20.0, num_sigma=2
        )

        # calculate cross-correlation coefficient of the two tomograms
        ccc = support.normalised_cross_correlation(tomo_1, tomo_2, mask=mask)

        print(
            "normalized cross correlation of two simulations of identical volume after downsampling both "
            "subtomograms 8 times = ",
            ccc,
        )
        self.assertGreater(
            ccc, 0.75, msg="correlation is not sufficient between simulations"
        )


if __name__ == "__main__":
    unittest.main()
