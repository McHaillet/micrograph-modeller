"""
Membrane geometrical structures

Depedencies: pyvista

Author: Marten Chaillet
"""

# essential
import numpy as np
import pyvista as pv
import scipy.ndimage as ndimage
from scipy.spatial import distance
from numba import jit
from potential import read_structure
from threadpoolctl import threadpool_info, threadpool_limits
from voltools.utils import translation_matrix, rotation_matrix


# ========================================== MEMBRANE PDB HELPER =======================================================
def boundary_box_from_pdb(filename):
    try:
        with open(filename, 'r') as pdb:
            line = pdb.readline()
            while line.split()[0] != 'CRYST1':
                # note: if we do not encounter CRYST1 in the file, we go to the except statement.
                line = pdb.readline()
        return float(line.split()[1]), float(line.split()[2]), float(line.split()[3])
    except Exception as e:
        print(e)
        raise Exception('Could not read pdb file.')


# ========================================= VECTOR CLASS AND Z ROT MATRIX ==============================================
class Vector:
    # Class can be used as both a 3d coordinate, and a vector
    # TODO SCIPY probably also has a vector class
    def __init__(self, coordinates, normalize=False):
        """
        Init vector with (x,y,z) coordinates, assumes (0,0,0) origin.
        """
        assert len(coordinates) == 3, 'Invalid axis list for a 3d vector, input does not contain 3 coordinates.'
        self._axis = np.array(coordinates)
        self._zero_vector = np.all(self._axis==0)
        if normalize:
            self.normalize()

    def get(self):
        """
        Return vector in numpy array.
        """
        return self._axis

    def show(self):
        """
        Print the vector.
        """
        print(self._axis)

    def copy(self):
        """
        Return a copy of the vector (also class Vector).
        """
        return Vector(self.get())

    def inverse(self):
        """
        Inverse the vector (in place).
        """
        return Vector(self._axis * -1)

    def cross(self, other):
        """
        Get cross product of self and other Vector. Return as new vector.
        """
        return Vector([self._axis[1] * other._axis[2] - self._axis[2] * other._axis[1],
                       self._axis[2] * other._axis[0] - self._axis[0] * other._axis[2],
                       self._axis[0] * other._axis[1] - self._axis[1] * other._axis[0]])

    def dot(self, other):
        """
        Return the dot product of vectors v1 and v2, of form (x,y,z).
        Dot product of two vectors is zero if they are perpendicular.
        """
        return self._axis[0] * other._axis[0] + self._axis[1] * other._axis[1] + self._axis[2] * other._axis[2]

    def magnitude(self):
        """
        Calculate the magnitude (length) of vector p.
        """
        return np.sqrt(np.sum(self._axis ** 2))

    def normalize(self):
        """
        Normalize self by dividing by magnitude.
        """
        if not self._zero_vector:
            self._axis = self._axis / self.magnitude()

    def angle(self, other, degrees=False):
        """
        Get angle between self and other.
        """
        # returns angle in radians
        if self._zero_vector or other._zero_vector:
            angle = 0
        else:
            angle = np.arccos(self.dot(other) / (self.magnitude() * other.magnitude()))
        if degrees:
            return angle * 180 / np.pi
        else:
            return angle

    def rotate(self, rotation_matrix):
        """
        Rotate the vector in place by the rotation matrix.
        """
        return Vector(np.dot(self._axis, rotation_matrix))

    def _get_orthogonal_unit_vector(self):
        """
        Get some orthogonal unit vector, multiple solutions are possible. Private method used in get rotation.
        """
        # A vector orthogonal to (a, b, c) is (-b, a, 0), or (-c, 0, a) or (0, -c, b).
        if self._zero_vector:
            return Vector([1, 0, 0])  # zero vector is orthogonal to any vector
        else:
            if self._axis[2] != 0:
                x, y = 1, 1
                z = (- 1 / self._axis[2]) * (x * self._axis[0] + y * self._axis[1])
            elif self._axis[1] != 0:
                x, z = 1, 1
                y = (- 1 / self._axis[1]) * (x * self._axis[0] + z * self._axis[2])
            else:
                y, z = 1, 1
                x = (- 1 / self._axis[0]) * (y * self._axis[1] + z * self._axis[2])
            orth = Vector([x, y, z])
            orth.normalize()
            np.testing.assert_allclose(self.dot(orth), 0, atol=1e-7, err_msg='something went wrong in finding ' \
                                                                             'perpendicular vector')
            return orth

    def get_rotation(self, other, as_affine_matrix=False):
        """
        Get rotation to rotate other vector onto self. Take the transpose of result to rotate self onto other.
        """
        if self._zero_vector or other._zero_vector:
            return np.identity(3)

        nself, nother = self.copy(), other.copy()
        nself.normalize()
        nother.normalize()

        if nself.dot(nother) > 0.99999:  # if the vectors are parallel
            return np.identity(3)  # return identity
        elif nself.dot(nother) < -0.99999:  # if the vectors are opposite
            axis = nself._get_orthogonal_unit_vector()  # return 180 along whatever axis
            angle = np.pi  # 180 degrees rotation around perpendicular vector
        else:
            axis = nself.cross(nother)
            axis.normalize()
            angle = nself.angle(nother)

        x, y, z = axis.get()
        c = np.cos(angle)
        s = np.sin(angle)
        t = 1.0 - c

        m00 = c + x * x * t
        m11 = c + y * y * t
        m22 = c + z * z * t

        tmp1 = x * y * t
        tmp2 = z * s
        m10 = tmp1 + tmp2
        m01 = tmp1 - tmp2
        tmp1 = x * z * t
        tmp2 = y * s
        m20 = tmp1 - tmp2
        m02 = tmp1 + tmp2
        tmp1 = y * z * t
        tmp2 = x * s
        m21 = tmp1 + tmp2
        m12 = tmp1 - tmp2

        mat = np.array([[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]])

        if as_affine_matrix:  # make 4x4
            out = np.identity(4)
            out[:3, :3] = mat
            mat = out

        return mat


def z_axis_rotation_matrix(angle):
    """
    Get a z-axis rotation matrix specified by angle in degrees.
    TODO scipy probably also has a function for this
    """
    m00 = np.cos(angle*np.pi/180)
    m01 = - np.sin(angle*np.pi/180)
    m10 = np.sin(angle*np.pi/180)
    m11 = np.cos(angle*np.pi/180)
    return np.array([[m00, m01, 0], [m10, m11, 0], [0, 0, 1]])


def rotation_matrix_to_affine_matrix(rotation_matrix):
    m = np.identity(4)
    m[0:3, 0:3] = rotation_matrix
    return m


# ======================================== SAMPLING POINTS ON ELLIPSE AND ELLIPSOID ====================================
def random_point_ellipsoid(a, b, c):
    # a,b, and c are paremeters of the ellipsoid
    # generating random (x,y,z) points on ellipsoid
    u = np.random.rand()
    v = np.random.rand()
    theta = u * 2.0 * np.pi
    phi = np.arccos(2.0 * v - 1.0) - np.pi / 2

    rx = a * np.cos(phi) * np.cos(theta)
    ry = b * np.cos(phi) * np.sin(theta)
    rz = c * np.sin(phi)

    return np.array([rx, ry, rz])


def random_point_ellipse(a, b):
    u = np.random.rand()
    theta = u * 2.0 * np.pi
    rx = a * np.cos(theta)
    ry = b * np.sin(theta)
    return np.array([rx, ry])


# ============================= HELPER FUNCTIONS FOR EQUILIBRATING ELLIPSE AND ELLIPSOIDS ==============================
@jit(nopython=True)
def get_root_ellipse(r0, z0, z1, g, maxiter=20):
    # use bisection method to find root ellipse
    n0 = r0 * z0
    s0, s1 = z1 - 1, 0 if g < 0 else np.sqrt(n0 ** 2 + z1 ** 2) - 1
    s = 0
    for _ in range(maxiter):
        s = (s0 + s1) / 2
        if s == s0 or s == s1:
            break
        ratio0, ratio1 = n0 / (s + r0), z1 / (s + 1)
        g = ratio0 ** 2 + ratio1 ** 2 - 1
        if g > 0:
            s0 = s
        elif g < 0:
            s1 = s
        else:
            break
    return s


@jit(nopython=True)
def get_root_ellipsoid(r0, r1, z0, z1, z2, g, maxiter=20):
    # use bisection method to find root ellipsoid
    n0, n1 = r0 * z0, r1 * z1
    s0, s1 = z2 - 1, 0 if g < 0 else np.sqrt(n0 ** 2 + n1 ** 2 + z2 ** 2) - 1
    s = 0
    for _ in range(maxiter):
        s = (s0 + s1) / 2
        if s == s0 or s == s1:
            break
        ratio0, ratio1, ratio2 = n0 / (s + r0), n1 / (s + r1), z2 / (s + 1)
        g = ratio0 ** 2 + ratio1 ** 2 + ratio2 ** 2 - 1
        if g > 0:
            s0 = s
        elif g < 0:
            s1 = s
        else:
            break
    return s


@jit(nopython=True)
def distance_point_ellipse_quadrant(e0, e1, y0, y1, maxiter=20, epsilon=0):
    """
    from https://www.geometrictools.com/Documentation/DistancePointEllipseEllipsoid.pdf
    e0 >= e1 > 0, y0 >= 0, y1 >= 0  (Y is in the first quadrant)
    """
    if y1 > epsilon:
        if y0 > epsilon:
            z0, z1 = y0 / e0, y1 / e1
            g = z0 ** 2 + z1 ** 2 - 1
            if g != 0:
                r0 = (e0 / e1) ** 2
                sbar = get_root_ellipse(r0, z0, z1, g, maxiter=maxiter)
                x0, x1 = r0 * y0 / (sbar + r0), y1 / (sbar + 1)
                distance = np.sqrt((x0 - y0) ** 2 + (x1 - y1) ** 2)
            else:
                x0, x1, distance = y0, y1, 0
        else:  # y0 == 0
            x0, x1, distance = 0, e1, abs(y1 - e1)
    else:  # y1 == 0
        numer0, denom0 = e0 * y0, e0 ** 2 - e1 ** 2
        if numer0 < denom0:
            xde0 = numer0 / denom0
            x0, x1 = e0 * xde0, e1 * np.sqrt(1 - xde0 ** 2)
            distance = np.sqrt((x0 - y0) ** 2 + x1 ** 2)
        else:
            x0, x1, distance = e0, 0, abs(y0 - e0)
    return x0, x1, distance  # return point, distance


@jit(nopython=True)
def distance_point_ellipsoid_octant(e0, e1, e2, y0, y1, y2, maxiter=20, epsilon=0):
    """
    from https://www.geometrictools.com/Documentation/DistancePointEllipseEllipsoid.pdf
    e0 >= e1 >= e2 > 0, y0 >= 0, y1 >= 0, y2 >= 0  (Y is in the first octant)
    """
    if y2 > epsilon:
        if y1 > epsilon:
            if y0 > epsilon:
                z0, z1, z2 = y0 / e0, y1 / e1, y2 / e2
                g = z0 ** 2 + z1 ** 2 + z2 ** 2 - 1
                if g != 0:
                    r0, r1 = (e0 / e2) ** 2, (e1 / e2) ** 2
                    sbar = get_root_ellipsoid(r0, r1, z0, z1, z2, g, maxiter=maxiter)
                    x0, x1, x2 = r0 * y0 / (sbar + r0), r1 * y1 / (sbar + r1), y2 / (sbar + 1)
                    distance = np.sqrt((x0 - y0) ** 2 + (x1 - y1) ** 2 + (x2 - y2) ** 2)
                else:
                    x0, x1, x2, distance = y0, y1, y2, 0
            else:  # y0 == 0
                x0 = 0
                x1, x2, distance = distance_point_ellipse_quadrant(e1, e2, y1, y2)
        else:  # y1 == 0
            if y0 > epsilon:
                x1 = 0
                x0, x2, distance = distance_point_ellipse_quadrant(e0, e2, y0, y2)
            else:  # y0 == 0
                x0, x1, x2, distance = 0, 0, e2, abs(y2 - e2)
    else:  # y2 == 0
        denom0, denom1 = e0 ** 2 - e2 ** 2, e1 ** 2 - e2 ** 2
        numer0, numer1 = e0 * y0, e1 * y1
        computed = False
        if numer0 < denom0 and numer1 < denom1:
            xde0, xde1 = numer0 / denom0 , numer1 / denom1
            xde0sqr, xde1sqr = xde0 ** 2, xde1 ** 2
            discr = 1 - xde0sqr - xde1sqr
            if discr > 0:
                x0, x1, x2 = e0 * xde0, e1 * xde1, e2 * np.sqrt(discr)
                distance = np.sqrt((x0 - y0) ** 2 + (x1 - y1) ** 2 + x2 ** 2)
                computed = True
        if not computed:
            x2 = 0
            x0, x1, distance = distance_point_ellipse_quadrant(e0, e1, y0, y1)
    return x0, x1, x2, distance


def place_back_to_ellipsoid(point, a, b, c, maxiter=20):
    # This is not a problem as we can anyway rotate the ellipsoid afterwards to place them in simulations,
    # so the directions of a, b and c do not matter.
    assert a >= b >= c > 0, "finding point currently only possible on ellipsoid with a >= b >= c > 0"

    # find rotation to move point to the first quadrant
    v1, v2 = Vector(point), Vector(abs(point))
    octant_rotation_matrix = v1.get_rotation(v2)  # this is the rotation of v2 onto v1

    # find rotation of ellipsoid parameters so that a >= b >= c > 0 holds
    x, y, z, _ = distance_point_ellipsoid_octant(a, b, c, *v2.get(), maxiter=maxiter)

    # return the intersection rotated to right octant
    return Vector([x, y, z]).rotate(octant_rotation_matrix).get()


def test_place_back_to_ellipsoid(size=11, a=10, b=3, c=1, iterations=20):
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    # xyz points between -a/b/c and a/b/c
    xx, yy, zz = map(int, [size*a, size*b, size*c])
    distances = np.zeros((xx, yy, zz))
    for i, x in enumerate(np.linspace(-a, a, xx)):
        for j, y in enumerate(np.linspace(-b, b, yy)):
            for k, z in enumerate(np.linspace(-c, c, zz)):
                point = np.array([x, y, z])
                ellipsoid_point = place_back_to_ellipsoid(point, a, b, c, maxiter=iterations)
                distances[i,j,k] = np.sqrt(distance.sqeuclidean(point, ellipsoid_point))
    # visualize
    slice_x = distances[xx//2,:,:]
    slice_y = distances[:,yy//2,:]
    slice_z = distances[:,:,zz//2]
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(slice_x)
    ax[1].imshow(slice_y)
    ax[2].imshow(slice_z)
    plt.show()
    return


# ============================================ TRIANGULATION ===========================================================
def sign(p1, p2, p3):
    """
    Determine on which side of the line formed by p2 and p3, p1 lays.
    """
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])


def point_array_sign(point_array, p2, p3):
    """
    Determine on which side of the line formed by p2 and p3, the points in point array lay.
    """
    return (point_array[:, 0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (point_array[:, 1] - p3[1])


class Triangle:
    def __init__(self, points, normal):
        assert isinstance(points[0], np.ndarray) and (len(points[0]) == 3), 'invalid triangle points provided'
        self.p1, self.p2, self.p3 = np.append(points[0], 1.), np.append(points[1], 1.), np.append(points[2], 1.)
        self.centroid = (1 / 3) * (points[0] + points[1] + points[2])
        if type(normal) is Vector:
            self.normal = normal
        else:
            self.normal = Vector(normal)

    def update_centroid(self):
        self.centroid = (1 / 3) * (self.p1[:3] + self.p2[:3] + self.p3[:3])

    def get_transformed_triangle(self, matrix):
        new_points = [np.dot(self.p1, matrix)[:3], np.dot(self.p2, matrix)[:3], np.dot(self.p3, matrix)[:3]]
        new_normal = np.dot(self.normal.get(), matrix[:3, :3])
        return Triangle(new_points, new_normal)

    def get_shifted_points(self, shift):
        return np.vstack([self.p1[:3] + shift, self.p2[:3] + shift, self.p3[:3] + shift])

    def get_points(self):
        return np.vstack([self.p1[:3], self.p2[:3], self.p3[:3]])

    def get_min(self):
        return np.array([min([p[i] for p in (self.p1, self.p2, self.p3)]) for i in [0, 1, 2]])

    def get_max(self):
        return np.array([max([p[i] for p in (self.p1, self.p2, self.p3)]) for i in [0, 1, 2]])

    def point_in_triangle(self, pt):
        """
        Return true if point lies inside triangle on the xy plane.
        """
        v1, v2, v3 = self.p1[:3], self.p2[:3], self.p3[:3]

        d1 = sign(pt, v1, v2)
        d2 = sign(pt, v2, v3)
        d3 = sign(pt, v3, v1)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)

    def point_array_in_triangle(self, point_array):
        """
        Return bool array with true if point lies inside triangle on the xy plane.
        """
        v1, v2, v3 = self.p1[:3], self.p2[:3], self.p3[:3]

        d1 = point_array_sign(point_array, v1, v2)
        d2 = point_array_sign(point_array, v2, v3)
        d3 = point_array_sign(point_array, v3, v1)

        # for numpy arrays
        has_neg = np.any([d1 < 0, d2 < 0, d3 < 0], axis=0)
        has_pos = np.any([d1 > 0, d2 > 0, d3 > 0], axis=0)

        return np.invert(np.all([has_neg, has_pos], axis=0))

    def display_triangle(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # <--- This is important for 3d plotting

        # first center triangle around origin
        triangle = self.get_points()
        normal = self.normal.get()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(triangle[:, 0], triangle[:, 1], triangle[:, 2])
        ax.quiver(*self.centroid, *normal, color='blue')
        if np.abs(triangle[:, 2].max() - triangle[:, 2].min()) < 0.001:
            ax.set_zlim3d(triangle[:, 2].mean() - 1, triangle[:, 2].mean() + 1)
        plt.show()


# =================== scipy cdist wrapper for iteratively updating points ==============================================
class DistanceMatrix:
    """
    Distance matrix wrapper for moving points and updating matrix with new points.
    """
    def __init__(self, points):
        self.matrix = distance.cdist(points, points, metric='sqeuclidean')
        # squared euclidean because we compare distance, saves computation time
        self.mean_distance = np.sqrt(self.matrix[self.matrix != 0].mean())
        self.upper = np.max(self.matrix)
        self.matrix[self.matrix == 0] = self.upper  # remove diagonal as its always 0

    def update(self, points, new_point_index):
        dist_update = np.sum((points - points[new_point_index]) ** 2, axis=1)  # squared euclidean (see above)
        dist_update[dist_update == 0] = self.upper  # remove point correlating with itself
        self.matrix[new_point_index, :] = dist_update  # update matrix
        self.matrix[:, new_point_index] = dist_update

    def shortest_distance(self):
        return np.unravel_index(self.matrix.argmin(), self.matrix.shape)


# ======================================= VESCILE FRAMEWORK AND SAMPLING ===============================================
class Vesicle:
    def __init__(self, radius, voxel_spacing):
        self.radius = radius
        # radius x10 to go to angstrom spacing of the vesicle, pdbs have angstrom units
        self.radii = sorted((x * self.radius for x in (np.random.uniform(0.8, 1.2), np.random.uniform(0.8, 1.2),
                                                       np.random.uniform(0.8, 1.2))), reverse=True)
        self.voxel_spacing = voxel_spacing
        self.reference_normal = Vector([.0, .0, 1.])  # dont see the point of making this an init param
        self.point_cloud = None
        self.framework = None  # list of triangles?
        self.occupation = None  # bool list of length n_triangles; true if membrane protein is placed

    def sample_ellipsoid_point_cloud(self, n):
        points = []
        for i in range(n):
            points.append(random_point_ellipsoid(*self.radii))
        self.point_cloud = np.vstack(points)

    def equilibrate_point_cloud(self, maxiter=10000, factor=0.01, display=False):
        dmatrix = DistanceMatrix(self.point_cloud)
        mean_dist = dmatrix.mean_distance

        for x in range(maxiter):
            if x % 1000 == 0:  # print progress
                print(f'equilibrator iteration {x}')

                if display:
                    self.display_point_cloud()

            # get the indices of the points that form the closest pair
            minp1, minp2 = dmatrix.shortest_distance()
            p1, p2 = self.point_cloud[minp1], self.point_cloud[minp2]

            # move closest pair away from each other
            scale = factor * (mean_dist / distance.euclidean(p1, p2))  # make scaling larger for tiny distances
            p1_new = p1 + scale * (p1 - p2)  # move p1 away from p2
            p2_new = p2 + scale * (p2 - p1)  # move p2 away from p1

            # use newton optimization to place the points back on the ellipsoid
            self.point_cloud[minp1] = place_back_to_ellipsoid(p1_new, *self.radii)
            self.point_cloud[minp2] = place_back_to_ellipsoid(p2_new, *self.radii)

            # update distance matrix with the new points
            dmatrix.update(self.point_cloud, minp1)
            dmatrix.update(self.point_cloud, minp2)

    def deform(self, strength):
        # generate random deform grids in xyz
        x_grid = (np.random.random((5, 5, 5)) - 0.5) * strength
        y_grid = (np.random.random((5, 5, 5)) - 0.5) * strength
        z_grid = (np.random.random((5, 5, 5)) - 0.5) * strength

        x, y, z = self.point_cloud[:, 0], self.point_cloud[:, 1], self.point_cloud[:, 2]
        x_sampling = ((x - x.min()) / (x.max() - x.min())) * 2
        y_sampling = ((y - y.min()) / (y.max() - y.min())) * 2
        z_sampling = ((z - z.min()) / (z.max() - z.min())) * 2
        sampling = np.vstack((x_sampling, y_sampling, z_sampling))

        x_deform = ndimage.map_coordinates(x_grid, sampling, order=2)
        y_deform = ndimage.map_coordinates(y_grid, sampling, order=2)
        z_deform = ndimage.map_coordinates(z_grid, sampling, order=2)

        self.point_cloud[:, 0] += x_deform
        self.point_cloud[:, 1] += y_deform
        self.point_cloud[:, 2] += z_deform

        print(f'deformed with 5x5 grid at strength {strength}')

    def generate_framework(self, alpha):
        """
        Pyvista can also directly generate an ellipsoid with:
            ellipsoid = pv.ParametricEllipsoid(10, 5, 5)
        This returns a surface as pyvista.PolyData and delaunay 3d should work directly on this
        """
        # points is a 3D numpy array (n_points, 3) coordinates of a sphere
        cloud = pv.PolyData(self.point_cloud)  # built-in pyvista plot: cloud.plot()

        # reconstructs the surface from a set of points on an assumed solid surface
        # for noise search for "pyvista perlin noise 3d"
        volume = cloud.delaunay_3d(alpha=alpha, progress_bar=True)
        shell = volume.extract_geometry()  # built-in pyvista plot: shell.plot()

        self.framework = []
        for i in range(shell.n_cells):  # make each surface cell a triangle in the framework
            self.framework.append(Triangle(shell.extract_cells(i).points, shell.cell_normals[i]))

    def sample_membrane(self, bilayer_pdb, cores=1):
        # READ THE STRUCTURE: membrane pdb should have solvent deleted at this point
        x_coordinates, y_coordinates, z_coordinates, \
            elements, b_factors, occupancies = map(np.array, read_structure(bilayer_pdb))
        z_coordinates -= z_coordinates.mean()  # center the layer in z
        n_atoms_box = len(elements)
        # get the periodic boundary box from the pdb file ==> membrane should be equilibrated with periodic boundaries
        x_bound, y_bound, z_bound = boundary_box_from_pdb(bilayer_pdb)

        membrane_x, membrane_y, membrane_z, membrane_e, membrane_b, membrane_o = [], [], [], [], [], []

        atom_count = 0

        user_apis = []
        for lib in threadpool_info():
            user_apis.append(lib['user_api'])
        if 'blas' in user_apis:
            print('BLAS in multithreading user apis, will try to limit number of threads for numpy.dot')

        for i_cell, triangle in enumerate(self.framework):

            # transform matrices
            mat_origin_shift = translation_matrix(triangle.centroid).T
            mat_ref_rot = self.reference_normal.get_rotation(triangle.normal, as_affine_matrix=True)
            mat_plane_rot = rotation_matrix((np.random.uniform(0, 360), 0, 0), rotation_order='rzxz')

            # get the transformed Triangle
            transf_triangle = triangle.get_transformed_triangle(np.dot(np.dot(mat_origin_shift, mat_ref_rot),
                                                                       mat_plane_rot))

            # find the shift so that triangle is in the positive quadrant of the xy axes
            shift_xy = np.append(transf_triangle.get_min()[:2], 0)  # already in xy plane so zshift should be 0
            mat_xy_shift = translation_matrix(shift_xy).T

            # construct the full transformation matrix to sample from the lipid layer
            matrix = np.dot(np.dot(mat_origin_shift, mat_ref_rot), np.dot(mat_plane_rot, mat_xy_shift))

            # transform triangle to get the x and y upper limits from it
            transf_triangle = triangle.get_transformed_triangle(matrix)
            xmax, ymax = transf_triangle.get_max()[:2]

            # find xy limits of the triangle and find increase compared to lipid boundary box
            xext = int(np.ceil(xmax / x_bound))
            yext = int(np.ceil(ymax / y_bound))

            atoms = np.empty((xext * yext * n_atoms_box, 4))  # numpy atom array with added dim for matrix mult
            newe = np.zeros(xext * yext * n_atoms_box, dtype='<U1')
            newb, newo = (np.zeros(xext * yext * n_atoms_box),) * 2

            # make periodic copies of membrane box in xy so the triangle fits
            for i in range(xext):
                for j in range(yext):
                    index = i * yext + j
                    # add ones to 4th coordinate position for affine transformation
                    atoms[index * n_atoms_box: (index + 1) * n_atoms_box, :] = np.array([x_coordinates + i * x_bound,
                                                                                         y_coordinates + j * y_bound,
                                                                                         z_coordinates,
                                                                                         np.ones(n_atoms_box)]).T
                    newe[index * n_atoms_box: (index + 1) * n_atoms_box] = elements.copy()
                    newb[index * n_atoms_box: (index + 1) * n_atoms_box] = b_factors.copy()
                    newo[index * n_atoms_box: (index + 1) * n_atoms_box] = occupancies.copy()

            # find all atoms that fall in the triangle
            locs = transf_triangle.point_array_in_triangle(atoms)
            atoms, newe, newb, newo = atoms[locs], newe[locs], newb[locs], newo[locs]

            if 'blas' in user_apis:
                with threadpool_limits(limits=cores, user_api='blas'):
                    ratoms = np.dot(atoms, np.linalg.inv(matrix))  # multiply the atoms with the matrix inverse
            else:
                ratoms = np.dot(atoms, np.linalg.inv(matrix))

            # add to the total atom lists
            membrane_x += list(ratoms[:, 0])
            membrane_y += list(ratoms[:, 1])
            membrane_z += list(ratoms[:, 2])
            membrane_e += list(newe)
            membrane_b += list(newb)
            membrane_o += list(newo)
            atom_count += len(newe)  # count total atoms

            # track the progress
            if i_cell % 500 == 0:
                print(f'At triangle {i_cell + 1} out of {len(self.framework)}. Current atom count is '
                      f'{atom_count}.')

        return membrane_x, membrane_y, membrane_z, membrane_e, membrane_b, membrane_o  # tuple

    def sample_protein(self, membrane_protein_pdb, n=1):
        """
        Sample n copies of the protein on the surface.

        Example code for placing only 1 protein in each triangle of the framework.
        for i in range(n):
            triangle = np.random.uniform(0, len(self.framework))
            while self.occupation[triangle]:
                triangle = np.random.uniform(0, len(self.framework))
            # place_protein()
            self.occupation[triangle] = True
        """
        pass

    def display_point_cloud(self, zlim=None):
        assert self.point_cloud is not None, 'point cloud not yet initialized, cannot display'

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # <--- This is important for 3d plotting

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.point_cloud[:, 0], self.point_cloud[:, 1], self.point_cloud[:, 2])
        if zlim is not None:
            ax.set_zlim3d(-zlim, zlim)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()
