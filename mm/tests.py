import unittest
import numpy as np
import voltools as vt
import support
from MicrographModeller import weighted_back_projection


def project(volume, angles):
    projections = []
    for a in angles:
        projections.append(vt.transform(volume, rotation=(0., a, 0.), rotation_order='rzyz',
                                        rotation_units='deg').mean(axis=2))
    return np.stack(projections, axis=2)


class ReconstructionTest(unittest.TestCase):
    def setUp(self):
        self.object = np.zeros((50, 50, 50), dtype=np.float32)
        self.object[18:23, 25:40, 33:39] = 1.
        self.tilt_angles = list(range(-60, 62, 2))
        self.projections = project(self.object, self.tilt_angles)

    def test(self):
        reconstruction, w = weighted_back_projection(self.projections, [(-t, 0, 0, 0, 1) for t in self.tilt_angles],
                                                     (50, 50, 50), (0, 0, 0), 1)
        support.write_mrc('/home/mchaillet/tests/projections.mrc', self.projections, 1)
        support.write_mrc('/home/mchaillet/tests/weighted.mrc', w, 1)
        support.write_mrc('/home/mchaillet/tests/original.mrc', self.object, 1)
        support.write_mrc('/home/mchaillet/tests/reconstruction.mrc', reconstruction, 1)

        original_normalised = (self.object - self.object.mean()) / self.object.std()
        reconstruction_normalised = (reconstruction - reconstruction.mean()) / reconstruction.std()

        ccc = (original_normalised * reconstruction_normalised).sum() / reconstruction_normalised.size
        self.assertGreater(ccc, 0.8)


if __name__ == '__main__':
    unittest.main()
