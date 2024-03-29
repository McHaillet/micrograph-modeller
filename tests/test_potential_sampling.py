import unittest
import numpy as np
from micrographmodeller import support, potential, utils


class TestElectrostaticPotential(unittest.TestCase):
    def setUp(self):
        self.param_pot = {
            "pdb": "test_data/3j9m.cif",
            "voxel_size": 5,
            "oversampling": 2,
            "solvent_exclusion": "gaussian",
            "absorption_contrast": True,
            "voltage": 300e3,
        }

    def test_base(self):
        # base method
        potential1 = potential.iasa_integration(
            self.param_pot["pdb"],
            voxel_size=self.param_pot["voxel_size"],
            oversampling=self.param_pot["oversampling"],
            solvent_exclusion=None,
            absorption_contrast=False,
            voltage=self.param_pot["voltage"],
        )

        ep = potential.ElectrostaticPotential(
            self.param_pot["pdb"],
            solvent_exclusion=None,
            absorption_contrast=False,
            voltage=self.param_pot["voltage"],
        )
        potential2 = ep.sample_to_box(
            voxel_size=self.param_pot["voxel_size"],
            oversampling=self.param_pot["oversampling"],
            center_coordinates_in_box=True,
            overhang=30,
            split=1,
            cores=4,
        )

        self.assertTrue(not np.iscomplexobj(potential1))
        self.assertTrue(not np.iscomplexobj(potential2))

        ccc = support.normalised_cross_correlation(potential1, potential2)
        print("base: ", ccc)
        self.assertGreater(ccc, 0.99)

        # base with solvent
        potential1 = potential.iasa_integration(
            self.param_pot["pdb"],
            voxel_size=self.param_pot["voxel_size"],
            oversampling=self.param_pot["oversampling"],
            solvent_exclusion=self.param_pot["solvent_exclusion"],
            absorption_contrast=self.param_pot["absorption_contrast"],
            voltage=self.param_pot["voltage"],
        )

        ep = potential.ElectrostaticPotential(
            self.param_pot["pdb"],
            solvent_exclusion=self.param_pot["solvent_exclusion"],
            absorption_contrast=self.param_pot["absorption_contrast"],
            voltage=self.param_pot["voltage"],
        )
        potential2 = ep.sample_to_box(
            voxel_size=self.param_pot["voxel_size"],
            oversampling=self.param_pot["oversampling"],
            center_coordinates_in_box=True,
            overhang=30,
            split=1,
            cores=4,
        )

        self.assertTrue(np.iscomplexobj(potential1))
        self.assertTrue(np.iscomplexobj(potential2))

        ccc = support.normalised_cross_correlation(potential1.real, potential2.real)
        print("base with solvent: ", ccc)
        self.assertGreater(ccc, 0.99)

        ccc = support.normalised_cross_correlation(potential1.imag, potential2.imag)
        print("base with solvent imag: ", ccc)
        self.assertGreater(ccc, 0.99)

    def test_splitting(self):
        # test whether splitting gives same results
        ep = potential.ElectrostaticPotential(
            self.param_pot["pdb"],
            solvent_exclusion=self.param_pot["solvent_exclusion"],
            absorption_contrast=self.param_pot["absorption_contrast"],
            voltage=self.param_pot["voltage"],
        )
        potential1 = ep.sample_to_box(
            voxel_size=self.param_pot["voxel_size"],
            oversampling=self.param_pot["oversampling"],
            center_coordinates_in_box=True,
            overhang=30,
            split=1,
            cores=4,
        )
        # support.write_mrc('./potential.mrc', potential1.real, 5)

        ep = potential.ElectrostaticPotential(
            self.param_pot["pdb"],
            solvent_exclusion=self.param_pot["solvent_exclusion"],
            absorption_contrast=self.param_pot["absorption_contrast"],
            voltage=self.param_pot["voltage"],
        )
        potential2 = ep.sample_to_box(
            voxel_size=self.param_pot["voxel_size"],
            oversampling=self.param_pot["oversampling"],
            center_coordinates_in_box=True,
            overhang=30,
            split=2,
            cores=4,
        )
        # support.write_mrc('./potential_split_cpu.mrc', potential2.real, 5)

        ccc = support.normalised_cross_correlation(potential1.real, potential2.real)
        print("splitting: ", ccc)
        self.assertGreater(ccc, 0.999)

        ccc = support.normalised_cross_correlation(potential1.imag, potential2.imag)
        print("splitting (imag): ", ccc)
        self.assertGreater(ccc, 0.999)

        ep = potential.ElectrostaticPotential(
            self.param_pot["pdb"],
            solvent_exclusion=self.param_pot["solvent_exclusion"],
            absorption_contrast=self.param_pot["absorption_contrast"],
            voltage=self.param_pot["voltage"],
        )
        potential3 = ep.sample_to_box(
            voxel_size=self.param_pot["voxel_size"],
            oversampling=self.param_pot["oversampling"],
            center_coordinates_in_box=True,
            overhang=30,
            split=3,
            cores=4,
        )
        # support.write_mrc('./potential_split_cpu.mrc', potential3.real, 5)

        ccc = support.normalised_cross_correlation(potential1.real, potential3.real)
        print("splitting: ", ccc)
        self.assertGreater(ccc, 0.999)

        ccc = support.normalised_cross_correlation(potential1.imag, potential3.imag)
        print("splitting (imag): ", ccc)
        self.assertGreater(ccc, 0.999)

    def test_solvent(self):
        # test whether splitting gives same results
        ep = potential.ElectrostaticPotential(
            self.param_pot["pdb"],
            solvent_exclusion="gaussian",
            absorption_contrast=self.param_pot["absorption_contrast"],
            voltage=self.param_pot["voltage"],
        )
        potential1 = ep.sample_to_box(
            voxel_size=self.param_pot["voxel_size"],
            oversampling=self.param_pot["oversampling"],
            center_coordinates_in_box=True,
            overhang=30,
            split=2,
            cores=4,
        )
        # support.write_mrc('./potential_gaussian.mrc', potential1.real, 5)

        ep = potential.ElectrostaticPotential(
            self.param_pot["pdb"],
            solvent_exclusion="masking",
            absorption_contrast=self.param_pot["absorption_contrast"],
            voltage=self.param_pot["voltage"],
        )
        potential2 = ep.sample_to_box(
            voxel_size=self.param_pot["voxel_size"],
            oversampling=self.param_pot["oversampling"],
            center_coordinates_in_box=True,
            overhang=30,
            split=2,
            cores=4,
        )
        # support.write_mrc('./potential_masking.mrc', potential2.real, 5)

        ccc = support.normalised_cross_correlation(potential1.real, potential2.real)
        print("splitting: ", ccc)
        self.assertGreater(ccc, 0.5)

    @unittest.skipUnless("gpu" in utils.AVAILABLE_DEVICES, "requires gpu")
    def test_gpu(self):
        # test whether splitting gives same results
        ep = potential.ElectrostaticPotential(
            self.param_pot["pdb"],
            solvent_exclusion=self.param_pot["solvent_exclusion"],
            absorption_contrast=self.param_pot["absorption_contrast"],
            voltage=self.param_pot["voltage"],
        )
        potential1 = ep.sample_to_box(
            voxel_size=self.param_pot["voxel_size"],
            oversampling=self.param_pot["oversampling"],
            center_coordinates_in_box=True,
            overhang=30,
            split=2,
            gpu_id=0,
        )
        # support.write_mrc('./potential_split_gpu.mrc', potential1.real, 5)

        ep = potential.ElectrostaticPotential(
            self.param_pot["pdb"],
            solvent_exclusion=self.param_pot["solvent_exclusion"],
            absorption_contrast=self.param_pot["absorption_contrast"],
            voltage=self.param_pot["voltage"],
        )
        potential2 = ep.sample_to_box(
            voxel_size=self.param_pot["voxel_size"],
            oversampling=self.param_pot["oversampling"],
            center_coordinates_in_box=True,
            overhang=30,
            split=2,
            cores=4,
        )

        ccc = support.normalised_cross_correlation(potential1.real, potential2.real)
        print("gpu: ", ccc)
        self.assertGreater(ccc, 0.999)

        ccc = support.normalised_cross_correlation(potential1.imag, potential2.imag)
        print("gpu (imag): ", ccc)
        self.assertGreater(ccc, 0.999)


if __name__ == "__main__":
    unittest.main()
