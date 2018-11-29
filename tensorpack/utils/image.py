import numpy as np
from . import logger
import cv2

__all__ = ['read_image']

try:
    import jpeg4py as jpeg
except ImportError:
    logger.warning('jpeg4py not installed, fallback to opencv for jpeg reading')
    jpeg = None


def read_image(file_name, bgr=True):
    """
    Reads jpeg by jpeg4py library. If jpeg4py is not installed or can't read the file, falls back to opencv imread()

    :param file_name: Image file
    :param bgr: Return BGR image instead of RGB
    :return: 3-channel image in RGB or BGR order
    """
    def read_opencv():
        image = cv2.imread(file_name, cv2.IMREAD_COLOR)
        if not bgr:
            image = image[..., ::-1]
        return image

    if jpeg:
        try:
            im = jpeg.JPEG(file_name).decode()
            if len(im.shape) == 2:
                im = np.stack((im,)*3, axis=-1)
            if bgr:
                im = im[..., ::-1]
            return im
        except jpeg.JPEGRuntimeError:
            # Fallback to read_opencv
            pass
    return read_opencv()
