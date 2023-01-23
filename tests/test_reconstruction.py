import unittest
import numpy as np
import voltools as vt
import micrographmodeller as mm


def project(volume, angles, in_plane_rotations, x_shifts, y_shifts):
    projections = []
    for a, r, x, y in zip(angles, in_plane_rotations, x_shifts, y_shifts):
        projections.append(vt.transform(volume, rotation=(0., a, r), rotation_order='rzyz',
                                        rotation_units='deg', translation=(x, y, 0)).mean(axis=2))
    return np.stack(projections, axis=2)


class TestReconstruction(unittest.TestCase):
    def setUp(self):
        self.object = np.zeros((50, 50, 50), dtype=np.float32)
        self.object[18:23, 25:40, 33:39] = 1.
        self.tilt_angles = list(range(-60, 62, 2))
        rng = np.random.default_rng(seed=1)
        self.in_plane_rotations = rng.random(len(self.tilt_angles)) * 10 - 5
        self.x_shifts = rng.random(len(self.tilt_angles)) * 2 - 1
        self.y_shifts = rng.random(len(self.tilt_angles)) * 2 - 1
        self.projections = project(self.object, self.tilt_angles, self.in_plane_rotations, self.x_shifts, self.y_shifts)

    def test(self):
        alignment = [(-t, -r, -x, -y, 1) for t, r, x, y in zip(self.tilt_angles, self.in_plane_rotations,
                                                               self.x_shifts, self.y_shifts)]
        reconstruction = mm.simulator.weighted_back_projection(self.projections, alignment,
                                                                        (50, 50, 50), (0, 0, 0), 1)
        # mm.support.write_mrc('./projections.mrc', self.projections, 1)
        # mm.support.write_mrc('./original.mrc', self.object, 1)
        # mm.support.write_mrc('./reconstruction.mrc', reconstruction, 1)

        self.assertGreater(mm.support.normalised_cross_correlation(self.object, reconstruction), 0.8)


if __name__ == '__main__':
    unittest.main()
