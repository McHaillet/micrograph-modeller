import numpy as np
import mrcfile

from micrographmodeller import microscope
from micrographmodeller import utils


DATATYPE_METAFILE = [
    ("DefocusU", "f4"),
    ("DefocusV", "f4"),
    ("DefocusAngle", "f4"),
    ("Voltage", "i4"),
    ("SphericalAberration", "f4"),
    ("AmplitudeContrast", "f4"),
    ("PhaseShift", "f4"),
    ("PixelSpacing", "f4"),
    ("MarkerDiameter", "i4"),
    ("TiltAngle", "f4"),
    ("RotationTheta", "f4"),
    ("InPlaneRotation", "f4"),
    ("TranslationX", "f4"),
    ("TranslationY", "f4"),
    ("TranslationZ", "f4"),
    ("Magnification", "f4"),
    ("Intensity", "f4"),
    ("ImageSize", "i4"),
    ("AcquisitionOrder", "i4"),
    ("FileName", "U1000"),
]
HEADER_METAFILE = ""
unitsMetaFile = [
    "um",
    "um",
    "deg",
    "kV",
    "micrographmodeller",
    "",
    "deg",
    "A",
    "A",
    "deg",
    "deg",
    "deg",
    "px",
    "px",
    "px",
    "",
    "",
    "px",
    "",
    "",
]
FMT_METAFILE = (
    "%11.6f %11.6f %6.2f %4d %6.2f %4.2f %11.6f %11.6f %4d "
    "%7.3f %7.3f %7.3f %6.2f %6.2f %6.2f %5.3f %5.3f %4d %3d %s"
)

for n, h in enumerate(DATATYPE_METAFILE):
    HEADER_METAFILE += "{} {}\n".format(
        h[0], "({})".format(unitsMetaFile[n]) * (unitsMetaFile[n] != "")
    )


def isheaderline(line):
    if (
        line.startswith("data_")
        or line.startswith("loop_")
        or line.startswith("_")
        or line.startswith("#")
    ):
        return True
    else:
        return False


def loadstar(filename, dtype="float32", usecols=None, skip_header=0, max_rows=None):
    # for dtype use DATATYPE_METAFILE to load a .meta file
    with open(filename, "r") as f:
        stop = 1e9 if max_rows is None else max_rows
        lines = [
            line
            for n, line in enumerate(f)
            if not isheaderline(line) and stop > n >= skip_header
        ]
        arr = np.genfromtxt(lines, dtype=dtype, usecols=usecols, max_rows=max_rows)
    return arr


def savestar(filename, arr, header="", fmt="", comments="#"):
    np.savetxt(filename, arr, comments=comments, header=header, fmt=fmt)


def read_mrc(filename, return_spacing_per_dim=False):
    # open the file in permissive mode, if there are big errors reading will fail
    with mrcfile.open(filename, permissive=True) as mrc:
        data = (
            mrc.data.transpose().copy()
        )  # copy after transpose to make memory C contiguous again
        vs = mrc.voxel_size
    if not return_spacing_per_dim:
        assert (
            vs["x"] == vs["y"] and vs["x"] == vs["z"]
        ), "spacing not equal in each dimension"
        vs = vs["x"]
    return data, vs


def write_mrc(filename, data, voxel_size, overwrite=True):
    assert not np.iscomplexobj(data), "cannot write complex objects to mrc"
    with mrcfile.new(filename, overwrite=overwrite) as mrc:
        mrc.set_data(
            data.astype(np.float32).transpose()
        )  # transpose because I use x y z indexing
        mrc.voxel_size = voxel_size


def fwhm_to_sigma(fwhm):
    return fwhm / (2 * np.sqrt(2 * np.log(2)))


def sigma_to_fwhm(sigma):
    return sigma * (2 * np.sqrt(2 * np.log(2)))


def hwhm_to_sigma(hwhm):
    return hwhm / (np.sqrt(2 * np.log(2)))


def sigma_to_hwhm(sigma):
    return sigma * (np.sqrt(2 * np.log(2)))


def create_gaussian_low_pass(shape, spacing, resolution, reduced=False, device="cpu"):
    """
    Create a 2D or 3D Gaussian low-pass filter with cutoff (or HWHM). This value will be converted to the proper
    sigma for the Gaussian function to acquire the desired cutoff.

    @param shape: shape tuple with x,y or x,y,z dimension
    @type  shape: L{tuple} -> (L{int},) * 3 or L{tuple} -> (L{int},) * 2
    @param hwhm: half width at half maximum for gaussian low-pass filter, the cutoff value for the filter
    @type  hwhm: L{int}
    @param center: center of the Gaussian, will be calculated if not provided
    @type  center: L{tuple} -> (L{int},) * 3 or L{tuple} -> (L{int},) * 2

    @return: sphere/circle in square volume/image, 3d or 2d array dependent on input shape
    @rtype:  L{np.ndarray}

    @author: Marten Chaillet
    """
    xp, _ = utils.get_array_module_from_device(device)
    assert len(shape) == 2 or len(shape) == 3, "filter can only be created in 2d or 3d"

    # 2 * spacing / resolution is cutoff in fourier space
    # then convert cutoff (hwhm) to sigma for gaussian function
    sigma_fourier = hwhm_to_sigma(2 * spacing / resolution)

    return xp.exp(
        -microscope.normalised_grid(shape, reduced=reduced, device=device) ** 2
        / (2 * sigma_fourier**2)
    )


def apply_fourier_filter(data, filter, human=True):
    """
    @param human: whether the filter is in human understandable form, i.e. zero frequency in size // 2 center
    """
    xp, _, _ = utils.get_array_module(data)

    if human:
        return xp.fft.ifftn(xp.fft.fftn(data) * xp.fft.ifftshift(filter)).real.astype(
            xp.float32
        )
    else:
        return xp.fft.ifftn(xp.fft.fftn(data) * filter).real.astype(xp.float32)


def reduce_resolution_fourier(data, spacing, resolution):
    """
    NOTE: MOVE FUNCTION TO agnostic.FILTER
    Apply scipy gaussian filter in fourier space.

    @param data: data to be filtered, either 2d or 3d array (however scipy will be able to handle higher
    dimensionality as well
    @type  data: L{np.ndarray}
    @param spacing: spacing of each pixel/voxel in relative units
    @type  spacing: L{float}
    @param resolution: desired resolution after filtering. maximal resolution is 2 * spacing, and thus resolution
    value of 2 * spacing will not filter the image
    @type  resolution: L{float}

    @return: filtered input, 2d or 3d array
    @rtype:  L{np.ndarray}

    @author: Marten Chaillet
    """
    xp, _, device = utils.get_array_module(data)

    gaussian_filter = create_gaussian_low_pass(
        data.shape, spacing, resolution, device=device
    )
    return apply_fourier_filter(data, gaussian_filter, human=True)


def reduce_resolution_real(data, spacing, resolution):
    """
    NOTE: MOVE FUNCTION TO agnostic.FILTER
    Apply scipy gaussian filter in real space.

    @param data: data to be filtered, either 2d or 3d array (however scipy will be able to handle higher
    dimensionality as well
    @type  data: L{np.ndarray}
    @param spacing: spacing of each pixel/voxel in relative units
    @type  spacing: L{float}
    @param resolution: desired resolution after filtering. maximal resolution is 2 * spacing, and thus resolution
    value of 2 * spacing will not filter the image
    @type  resolution: L{float}

    @return: filtered data, 2d or 3d array
    @rtype:  L{np.ndarray}

    @author: Marten Chaillet
    """
    _, ndimage, _ = utils.get_array_module(data)

    # here it needs to be fwhm to sigma, while in fourier space hwhm to sigma
    return ndimage.gaussian_filter(
        data, sigma=fwhm_to_sigma(resolution / (2 * spacing))
    )


def gradient_image(size, factor, angle=0, center_shift=0, device="cpu"):
    """
    Creates an image with a gradient of values rotated along angle. Factor determines the strength of the gradient.

    @param size: size of the image, x and y size are equal
    @type  size: L{int}
    @param factor: strength of gradient, value between 0 and 1, where 0 is no gradient and 1 a gradient from 0 to 2
    @type  factor L{float}
    @param angle: angle to rotate the gradient by
    @type  angle: L{float}
    @param center_shift: whether to shift the gradient from perfect center, if no shift applied center value will
    always be equal to 1
    @type  center_shift: L{float}

    @return: image, a 2d array of floats
    @rtype:  L{np.ndarray}

    @author: Marten Chaillet
    """
    xp, ndimage, _ = utils.get_array_module_from_device(device)

    max_rotation_radius = (size / 2) / xp.cos(45 * xp.pi / 180)
    extension = int(xp.ceil(max_rotation_radius - size / 2))
    left = 1 - factor
    right = 1 + factor
    step = (right - left) / size
    values = xp.arange(
        left - extension * step + center_shift * step,
        right + extension * step + center_shift * step,
        step,
    )
    image = xp.repeat(values[xp.newaxis, :], size + 2 * extension, axis=0)
    return ndimage.rotate(image, angle, reshape=False)[
        extension : size + extension, extension : size + extension
    ]


def create_circle(shape, radius=-1, sigma=0, center=None, device="cpu"):
    """
    Create a circle with radius in an image of shape.

    @param shape: shape of the image
    @type  shape: L{tuple} -> (L{int},) * 2
    @param radius: radius of the circle
    @type  radius: L{float}
    @param sigma: smooth gaussian edge of circle
    @type  sigma: L{float}
    @param center: center of the circle in the image
    @type  center: L{tuple} -> (L{int},) * 2

    @return: image with circle, 2d array of floats
    @rtype:  L{np.ndarray}

    @author: Marten Chaillet
    """
    xp, _ = utils.get_array_module_from_device(device)
    assert len(shape) == 2

    if center is None:
        center = [shape[0] / 2, shape[1] / 2]
    if radius == -1:
        radius = xp.min(shape) / 2

    sphere = xp.zeros(shape)
    [x, y] = xp.mgrid[0 : shape[0], 0 : shape[1]]
    r = xp.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    sphere[r <= radius] = 1

    if sigma > 0:
        ind = xp.logical_and(r > radius, r < radius + 2 * sigma)
        sphere[ind] = xp.exp(-(((r[ind] - radius) / sigma) ** 2) / 2)

    return sphere


def create_ellipsoid(size, mj, mn1, mn2, smooth=0, cutoff_SD=3, device="cpu"):
    """
    Generate an ellipse defined by 3 radii along x,y,z - parameters mj, mn1, mn2. Ellipse is generated in a square
    volume with each dimension has same size.

    @param size: length of dimensions
    @type  size: L{int}
    @param mj: major radius
    @type  mj: L{float}
    @param mn1: minor radius 1
    @type  mn1: L{float}
    @param mn2: minor radius 2
    @type  mn2: L{float}
    @param smooth: gaussian smoothing of ellips edges, where smooth is the sigma of gaussian function
    @type  smooth: L{float}
    @param cutoff_SD: number of standard deviations to determine gaussian smoothing to
    @type  cutoff_SD: L{int}

    @return: square volume with ellipse, 3d array of floats
    @rtype:  L{np.ndarray}

    @author: Marten Chaillet
    """
    xp, _ = utils.get_array_module_from_device(device)

    X, Y, Z = xp.meshgrid(xp.arange(size / 1), xp.arange(size / 1), xp.arange(size / 1))

    X -= size / 2 - 0.5
    Y -= size / 2 - 0.5
    Z -= size / 2 - 0.5

    R = xp.sqrt((X / mj) ** 2 + (Y / mn1) ** 2 + (Z / mn2) ** 2)

    # print(R.max(), R.min())

    out = xp.zeros((size, size, size), dtype=xp.float32)
    out[R <= 1] = 1

    if smooth:
        R2 = R.copy()
        R2[R <= 1] = 1
        sphere = xp.exp(-1 * ((R2 - 1) / smooth) ** 2)
        sphere[sphere <= xp.exp(-(cutoff_SD**2) / 2.0)] = 0
        out = sphere

    return out


def create_sphere(size, radius=-1, sigma=0, num_sigma=2, center=None, device="cpu"):
    """Create a 3D sphere volume.
    @param size: size of the resulting volume.
    @param radius: radius of the sphere inside the volume.
    @param sigma: sigma of the Gaussian.
    @param center: center of the sphere.
    @return: sphere inside a volume.
    """
    xp, _ = utils.get_array_module_from_device(device)

    if size.__class__ == float or len(size) == 1:
        size = (size, size, size)
    assert len(size) == 3
    if center is None:
        center = [size[0] // 2, size[1] // 2, size[2] // 2]
    if radius == -1:
        radius = min(size) // 2

    sphere = xp.zeros(size, dtype=xp.float32)
    [x, y, z] = xp.mgrid[0 : size[0], 0 : size[1], 0 : size[2]]
    r = xp.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)
    sphere[r <= radius] = 1

    if sigma > 0:
        ind = xp.logical_and(r > radius, r < radius + num_sigma * sigma)
        sphere[ind] = xp.exp(-(((r[ind] - radius) / sigma) ** 2) / 2)

    return sphere


def bandpass_mask(shape, low=0, high=-1, device="cpu"):
    """
    Return 2d bandpass mask in shape. Mask is created by subtracting a circle with radius low from a circle with
    radius high.

    @param shape: shape of image
    @type  shape: L{tuple} -> (L{int},) * 2
    @param low: inner radius of band
    @type  low: L{float}
    @param high: outer radius of band
    @type  high: L{float}

    @return: an image with bandpass, 2d array of ints
    @rtype:  L{np.ndarray}

    @author: Marten Chaillet
    """
    assert low >= 0, "lower limit must be >= 0"

    if high == -1:
        high = min(shape) / 2
    assert low < high, "upper bandpass must be > than lower limit"

    if low == 0:
        mask = create_circle(shape, high, sigma=0, device=device)
    else:
        mask = create_circle(shape, high, sigma=0, device=device) - create_circle(
            shape, low, sigma=0, device=device
        )

    return mask


def mean_under_mask(volume, mask):
    """
    Determines the mean value under a mask
    @param volume: The volume
    @type volume:  L{np.ndarray}
    @param mask:  The mask
    @type mask:  L{np.ndarray}

    @return: A scalar
    @rtype: L{float}
    """
    return (volume * mask).sum() / mask.sum()


def bin_volume(potential, factor):
    """
    Bin the data volume (potential) factor times.

    @param potential: data volume, 3d array
    @type  potential: L{np.ndarray}
    @param factor: integer multiple of 1, number of times to bin (or downsample)
    @type  factor: L{int}

    @return: downsampled data, 3d array
    @rtype:  L{np.ndarray}

    @author: Marten Chaillet
    """
    assert type(factor) is int and factor >= 1, print(
        "non-valid binning factor, should be integer above 1"
    )

    if factor == 1:
        return potential

    size = potential.shape
    s = [(x % factor) // 2 for x in size]
    d = [(x % factor) % 2 for x in size]
    # print(size, s, d)
    # s = (potential.shape[0] % factor) // 2
    # d = (potential.shape[0] % factor) % 2

    potential = potential[
        s[0] : size[0] - s[0] - d[0],
        s[1] : size[1] - s[1] - d[1],
        s[2] : size[2] - s[2] - d[2],
    ]
    # potential = potential[s:potential.shape[0] - s - d, s:potential.shape[0] - s - d, s:potential.shape[0] - s - d]

    size = potential.shape if potential.shape != size else size
    # ds = int(potential.shape[0]//factor)
    ds = [int(x // factor) for x in size]
    # image_size = potential.shape[0]

    # binned = potential.reshape(ds, image_size // ds,
    #                            ds, image_size // ds, ds, image_size // ds).mean(-1).mean(1).mean(-2)
    binned = (
        potential.reshape(
            ds[0], size[0] // ds[0], ds[1], size[1] // ds[1], ds[2], size[2] // ds[2]
        )
        .mean(-1)
        .mean(1)
        .mean(-2)
    )

    return binned


def add_correlated_noise(noise_size, dim, device="cpu"):
    """
    Add correlated noise to create density deformations.

    @param noise_size: strength of the correlation in noise, in number of pixels
    @type  noise_size: L{int}
    @param dim: dimension of the volume to create the correlated noise in
    @type  dim: L{int}

    @return: volume with noise, 3d array of floats
    @rtype:  L{np.ndarray}

    @author: Marten Chaillet
    """
    xp, _ = utils.get_array_module_from_device(device)

    if noise_size == 0:
        # 0 value will not work
        noise_size = 1

    noise_no_norm = abs(
        xp.fft.ifftn(
            xp.fft.fftshift(
                xp.fft.fftn(xp.random.random((noise_size, noise_size, noise_size)))
            ),
            [dim] * 3,
        )
    )
    noise = 0.2 * noise_no_norm / abs(noise_no_norm).max()

    return 1 + (noise - noise.mean())


def paste_in_center(volume, volume2):
    l, l2 = len(volume.shape), len(volume.shape)
    assert l == l2, "not same number of dims"
    for i in range(l):
        assert volume.shape[i] <= volume2.shape[i]

    if len(volume.shape) == 3:
        sx, sy, sz = volume.shape
        SX, SY, SZ = volume2.shape
        if SX <= sx:
            volume2[:, :, :] = volume[
                sx // 2 - SX // 2 : sx // 2 + SX // 2 + SX % 2,
                sy // 2 - SY // 2 : sy // 2 + SY // 2 + SY % 2,
                sz // 2 - SZ // 2 : sz // 2 + SZ // 2 + SZ % 2,
            ]
        else:
            volume2[
                SX // 2 - sx // 2 : SX // 2 + sx // 2 + sx % 2,
                SY // 2 - sy // 2 : SY // 2 + sy // 2 + sy % 2,
                SZ // 2 - sz // 2 : SZ // 2 + sz // 2 + sz % 2,
            ] = volume
        return volume2

    if len(volume.shape) == 2:
        sx, sy = volume.shape
        SX, SY = volume2.shape
        volume2[
            SX // 2 - sx // 2 : SX // 2 + sx // 2 + sx % 2,
            SY // 2 - sy // 2 : SY // 2 + sy // 2 + sy % 2,
        ] = volume
        return volume2


def ramp_filter(size, device="cpu"):
    """
    rampFilter: Generates the weighting function required for weighted backprojection - y-axis is tilt axis
    @param size: tuple of ints (x, y)
    @param device: cpu or gpu
    @return: filter volume
    """
    xp, _ = utils.get_array_module_from_device(device)

    # maximum frequency
    crowther = size[0] // 2

    # create a line increasing from center 0 to edge 1
    ramp_line = xp.abs(xp.arange(-size[0] // 2, size[0] // 2)) / crowther

    # 1 should be max
    ramp_line[ramp_line > 1] = 1

    # extend in y
    return xp.column_stack(
        [
            ramp_line,
        ]
        * size[1]
    )


def taper_mask(size, width, device="cpu"):
    """
    taper edges of image (or volume) with cos function
    @param size: shape as tuple of two ints (x, y)
    @param width: width of edge
    @param device: cpu or gpu
    @return: taper mask
    """
    xp, _ = utils.get_array_module_from_device(device)

    width = int(round(width))
    val = xp.cos(xp.arange(1, width + 1) * xp.pi / (2.0 * width))
    taper_x = xp.ones((size[0]), dtype=xp.float32)
    taper_y = xp.ones((size[1]))
    taper_x[:width] = val[::-1]
    taper_x[-width:] = val
    taper_y[:width] = val[::-1]
    taper_y[-width:] = val
    x, y = xp.meshgrid(taper_y, taper_x)
    return x * (x < y) + y * (y <= x)


def normalise_under_mask(array, mask):
    mean = (array * mask).sum() / mask.sum()
    std = ((array**2 * mask).sum() / mask.sum() - mean**2) ** 0.5
    return (array - mean) / std


def normalised_cross_correlation(array1, array2, mask=None):
    if mask is None:
        a = (array1 - array1.mean()) / array1.std()
        b = (array2 - array2.mean()) / array2.std()
        return (a * b).sum() / a.size
    else:
        a = normalise_under_mask(array1, mask)
        b = normalise_under_mask(array2, mask)
        return (a * b * mask).sum() / mask.sum()
