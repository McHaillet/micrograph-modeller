import GPUtil
import numpy as np
import scipy.ndimage

try:
    import cupy as cp
    import cupyx.scipy.ndimage
except ImportError:
    pass


def get_available_devices():

    available_devices = ["cpu"]

    # check if cupy is installed
    try:
        import cupy

        # add auto gpu
        available_devices.append("gpu")

        # get all available gpus
        gpu_ids = GPUtil.getAvailable(limit=100, includeNan=True)

        # add gpus to list of available devices
        for i in gpu_ids:
            available_devices.append(f"gpu:{i}")

    except ImportError:
        print(
            'Warning: cupy is not found. Therefore, the only available device is "cpu".\n'
            "Please install cupy>=10.6.0:\npip install cupy>=10.6.0"
        )

    return available_devices


AVAILABLE_DEVICES = get_available_devices()


def get_array_module(array):
    if type(array) is np.ndarray:
        return np, scipy.ndimage, "cpu"
    else:  # if not a np array, it must be cupy
        return cp, cupyx.scipy.ndimage, "gpu:" + str(array.device.id)


def get_array_module_from_device(device):
    if device not in AVAILABLE_DEVICES:
        raise ValueError(
            f"Unknown device ({device}), must be one of {AVAILABLE_DEVICES}"
        )

    if "cpu" in device:
        return np, scipy.ndimage
    else:
        return cp, cupyx.scipy.ndimage


# parses device string and switches cupy to specific id
def switch_to_device(device):
    if device not in AVAILABLE_DEVICES:
        raise ValueError(
            f"Unknown device ({device}), must be one of {AVAILABLE_DEVICES}"
        )

    # if id provided
    if device[4:]:
        cp.cuda.Device(int(device[4:])).use()


class StructureModificationError(Exception):
    """Class for excepting pdb modification error."""
    pass
