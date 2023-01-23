import unittest
import numpy as np
import micrographmodeller as mm


class TestElectrostaticPotential(unittest.TestCase):
    def setUp(self):
        self.param_pot = {
            'pdb': 'test_data/3j9m.cif',
            'voxel_size': 5,
            'oversampling': 2,
            'solvent_exclusion': 'gaussian',
            'absorption_contrast': True,
            'voltage': 300e3
        }

    @unittest.skip("only splitting")
    def test_base(self):
        # base method
        potential1 = mm.potential.iasa_integration(self.param_pot['pdb'],
                                                       voxel_size=self.param_pot['voxel_size'],
                                                       oversampling=self.param_pot['oversampling'],
                                                       solvent_exclusion=None,
                                                       absorption_contrast=False,
                                                       voltage=self.param_pot['voltage'])

        ep = mm.potential.ElectrostaticPotential(self.param_pot['pdb'],
                                                 solvent_exclusion=None,
                                                 absorption_contrast=False,
                                                 voltage=self.param_pot['voltage'])
        potential2 = ep.sample_to_box(voxel_size=self.param_pot['voxel_size'],
                                     oversampling=self.param_pot['oversampling'],
                                     center_coordinates_in_box=True, overhang=30, split=1, cores=4)

        self.assertTrue(not np.iscomplexobj(potential1))
        self.assertTrue(not np.iscomplexobj(potential2))

        ccc = mm.support.normalised_cross_correlation(potential1, potential2)
        print('base: ', ccc)
        self.assertGreater(ccc, 0.99)

        # base with solvent
        potential1 = mm.potential.iasa_integration(self.param_pot['pdb'],
                                                   voxel_size=self.param_pot['voxel_size'],
                                                   oversampling=self.param_pot['oversampling'],
                                                   solvent_exclusion=self.param_pot['solvent_exclusion'],
                                                   absorption_contrast=self.param_pot['absorption_contrast'],
                                                   voltage=self.param_pot['voltage'])

        ep = mm.potential.ElectrostaticPotential(self.param_pot['pdb'],
                                                 solvent_exclusion=self.param_pot['solvent_exclusion'],
                                                 absorption_contrast=self.param_pot['absorption_contrast'],
                                                 voltage=self.param_pot['voltage'])
        potential2 = ep.sample_to_box(voxel_size=self.param_pot['voxel_size'],
                                      oversampling=self.param_pot['oversampling'],
                                      center_coordinates_in_box=True, overhang=30, split=1, cores=4)

        self.assertTrue(np.iscomplexobj(potential1))
        self.assertTrue(np.iscomplexobj(potential2))

        ccc = mm.support.normalised_cross_correlation(potential1.real, potential2.real)
        print('base with solvent: ', ccc)
        self.assertGreater(ccc, 0.99)

        ccc = mm.support.normalised_cross_correlation(potential1.imag, potential2.imag)
        print('base with solvent imag: ', ccc)
        self.assertGreater(ccc, 0.99)

    def test_splitting(self):
        # test whether splitting gives same results
        ep = mm.potential.ElectrostaticPotential(self.param_pot['pdb'],
                                                 solvent_exclusion=self.param_pot['solvent_exclusion'],
                                                 absorption_contrast=self.param_pot['absorption_contrast'],
                                                 voltage=self.param_pot['voltage'])
        potential1 = ep.sample_to_box(voxel_size=self.param_pot['voxel_size'],
                                      oversampling=self.param_pot['oversampling'],
                                      center_coordinates_in_box=True, overhang=30, split=1, cores=4)
        mm.support.write_mrc('./potential1.mrc', potential1.real, 5)

        ep = mm.potential.ElectrostaticPotential(self.param_pot['pdb'],
                                                 solvent_exclusion=self.param_pot['solvent_exclusion'],
                                                 absorption_contrast=self.param_pot['absorption_contrast'],
                                                 voltage=self.param_pot['voltage'])
        potential2 = ep.sample_to_box(voxel_size=self.param_pot['voxel_size'],
                                      oversampling=self.param_pot['oversampling'],
                                      center_coordinates_in_box=True, overhang=30, split=2, cores=4)
        mm.support.write_mrc('./potential2.mrc', potential2.real, 5)

        ccc = mm.support.normalised_cross_correlation(potential1.real, potential2.real)
        print('splitting: ', ccc)
        self.assertGreater(ccc, 0.99)

        ccc = mm.support.normalised_cross_correlation(potential1.imag, potential2.imag)
        print('splitting (imag): ', ccc)
        self.assertGreater(ccc, 0.99)

    @unittest.skip("only splitting")
    def test_solvent(self):
        # test whether splitting gives same results
        ep = mm.potential.ElectrostaticPotential(self.param_pot['pdb'],
                                                 solvent_exclusion='gaussian',
                                                 absorption_contrast=self.param_pot['absorption_contrast'],
                                                 voltage=self.param_pot['voltage'])
        potential1 = ep.sample_to_box(voxel_size=self.param_pot['voxel_size'],
                                      oversampling=self.param_pot['oversampling'],
                                      center_coordinates_in_box=True, overhang=30, split=2, cores=4)

        ep = mm.potential.ElectrostaticPotential(self.param_pot['pdb'],
                                                 solvent_exclusion='masking',
                                                 absorption_contrast=self.param_pot['absorption_contrast'],
                                                 voltage=self.param_pot['voltage'])
        potential2 = ep.sample_to_box(voxel_size=self.param_pot['voxel_size'],
                                      oversampling=self.param_pot['oversampling'],
                                      center_coordinates_in_box=True, overhang=30, split=2, cores=4)

        ccc = mm.support.normalised_cross_correlation(potential1.real, potential2.real)
        print('splitting: ', ccc)
        self.assertGreater(ccc, 0.9)

    @unittest.skipUnless('gpu' in mm.utils.AVAILABLE_DEVICES, 'requires gpu')
    def test_gpu(self):
        # test whether splitting gives same results
        ep = mm.potential.ElectrostaticPotential(self.param_pot['pdb'],
                                                 solvent_exclusion='masking',
                                                 absorption_contrast=self.param_pot['absorption_contrast'],
                                                 voltage=self.param_pot['voltage'])
        potential1 = ep.sample_to_box(voxel_size=self.param_pot['voxel_size'],
                                      oversampling=self.param_pot['oversampling'],
                                      center_coordinates_in_box=True, overhang=30, split=2, gpu_id=0)

        ep = mm.potential.ElectrostaticPotential(self.param_pot['pdb'],
                                                 solvent_exclusion='masking',
                                                 absorption_contrast=self.param_pot['absorption_contrast'],
                                                 voltage=self.param_pot['voltage'])
        potential2 = ep.sample_to_box(voxel_size=self.param_pot['voxel_size'],
                                      oversampling=self.param_pot['oversampling'],
                                      center_coordinates_in_box=True, overhang=30, split=2, cores=4)

        ccc = mm.support.normalised_cross_correlation(potential1.real, potential2.real)
        print('gpu: ', ccc)
        self.assertGreater(ccc, 0.99)

        ccc = mm.support.normalised_cross_correlation(potential1.imag, potential2.imag)
        print('gpu (imag): ', ccc)
        self.assertGreater(ccc, 0.99)


if __name__ == '__main__':
    unittest.main()
