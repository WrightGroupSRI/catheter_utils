"""Catheter localization algorithms. Take projection data and produce
coordinate estimates."""

import logging
from itertools import chain

import numpy
import scipy.ndimage
import scipy.optimize

logger = logging.getLogger(__name__)


def peak(fs, xs):
    """Localize the catheter in the projection direction by finding the max
    signal."""
    return xs[numpy.argmax(fs)]


def centroid(fs, xs):
    """Localize the catheter in the projection direction by finding the
    centroid of the signal."""
    return numpy.trapz(fs * xs) / numpy.trapz(fs)


def centroid_around_peak(fs, xs, window_radius=3.5):
    """Localize the catheter in the projection direction by finding the
    centroid of the signal in a window centered around the peak."""
    window = numpy.abs(xs - peak(fs, xs)) < window_radius
    return centroid(fs[window], xs[window])


def iterative_weighted_centroid(fs, xs, weighting, tol=1e-6, max_iter=32):
    """Localize the catheter in the projection direction by iteratively
    finding the centroid using a weighting function centered at the
    previous location."""

    x0 = peak(fs, xs)

    for itr in range(max_iter):
        x = x0
        x0 = centroid(fs * weighting(xs - x), xs)

        if abs(x0 - x) <= tol:
            logger.debug("converged in %s iterations", itr)
            break
    else:
        logger.debug("maximum iterations reached")

    return x0


def png_density(xs, width=3, sigma=0.5):
    """'Peak Normed Gaussian' is a function used for iterative catheter
    localization. It is 1 in the range [-width, width] before decaying as a
    gaussian."""
    ws = numpy.abs(xs)
    ws[ws < width] = width
    return numpy.exp(-0.5 * (ws - width) ** 2 / (sigma ** 2))


def png(fs, xs, width=3, sigma=0.5, tol=1e-6, max_iter=32):
    """'Peak Normed Gaussian' localizes the catheter in the projection
    direction by iteratively computing the 'png' weighted centroid."""

    def weights(ws):
        return png_density(ws, width, sigma)

    return iterative_weighted_centroid(fs, xs, weights, tol=tol, max_iter=max_iter)





class XYZCoordinates:
    """Convert between XYZ projection ("sri") projection and world
    coordinates."""

    @staticmethod
    def projection_to_world(x):
        """Transform from `projection` coordinates to `world`
        coordinates."""
        return x

    @staticmethod
    def world_to_projection(u):
        """Transform from `world` coordinates to `projection`
        coordinates."""
        return u


class HadamardCoordinates:
    """Convert between Hadamard projection coordinates ("fh") and world
    coordinates."""

    # s2p, 3 world coordinates -> 4 projection coordinates
    # Each 'projection' coordinate is calculated by projecting
    # the world coordinate onto the corresponding direction
    # unit vector.
    _BACKWARD = (3**-0.5) * numpy.array([
        [-1, -1, -1],
        [1, 1, -1],
        [1, -1, 1],
        [-1, 1, 1]
    ])

    # p2s, 4 projection coordinates -> 3 world coordinates
    # Least squares solution provided by pinv.
    _FORWARD = numpy.linalg.pinv(_BACKWARD)

    @classmethod
    def projection_to_world(cls, x):
        """Transform from `projection` coordinates to `world`
        coordinates."""
        return cls._FORWARD @ x

    @classmethod
    def world_to_projection(cls, u):
        """Transform from `world` coordinates to `projection`
        coordinates."""
        return cls._BACKWARD @ u


def localize_coil(data, localizer, localizer_args=None, localizer_kwargs=None):
    if localizer_args is None:
        localizer_args = []
    if localizer_kwargs is None:
        localizer_kwargs = {}

    if len(data) == 3:
        coordinate_system = XYZCoordinates()
    elif len(data) == 4:
        coordinate_system = HadamardCoordinates()
    else:
        raise ValueError("Expected 3 or 4 projection directions")

    return coordinate_system.projection_to_world(numpy.array([
        localizer(fs, xs, *localizer_args, **localizer_kwargs) for fs, xs in data
    ]))


def localize_catheter(distal_data, proximal_data, localizer, localizer_args=None, localizer_kwargs=None):
    distal = localize_coil(distal_data, localizer, localizer_args, localizer_kwargs)
    proximal = localize_coil(proximal_data, localizer, localizer_args, localizer_kwargs)
    return distal, proximal


def joint_iterative_weighted_centroid(distal_data, proximal_data, geometry, weighting, tol=1e-6, max_iter=256):

    if len(distal_data) == 3 and len(proximal_data) == 3:
        coordinate_system = XYZCoordinates()
    elif len(distal_data) == 4 and len(proximal_data) == 4:
        coordinate_system = HadamardCoordinates()
    else:
        raise ValueError("Expected 3 or 4 projection directions")

    d0 = coordinate_system.projection_to_world(numpy.array([
        peak(fs, xs) for fs, xs in distal_data
    ]))
    p0 = coordinate_system.projection_to_world(numpy.array([
        peak(fs, xs) for fs, xs in proximal_data
    ]))

    for itr in range(max_iter):
        d1 = d0
        d = coordinate_system.world_to_projection(d1)
        d0 = coordinate_system.projection_to_world(
            numpy.array([centroid(fs * weighting(xs - x), xs) for (fs, xs), x in zip(distal_data, d)])
        )

        p1 = p0
        p = coordinate_system.world_to_projection(p1)
        p0 = coordinate_system.projection_to_world(
            numpy.array([centroid(fs * weighting(xs - x), xs) for (fs, xs), x in zip(proximal_data, p)])
        )

        fit = geometry.fit_from_coils_mse(d0, p0)
        d0 = fit.distal.reshape(-1)
        p0 = fit.proximal.reshape(-1)

        if numpy.linalg.norm(d0 - d1) + numpy.linalg.norm(p0 - p1) <= tol:
            logger.debug("converged in %s iterations", itr)
            break
    else:
        logger.debug("maximum iterations reached")

    return d0, p0


def jpng(distal_data, proximal_data, geometry, width=3, sigma=0.5, tol=1e-6, max_iter=256):

    def weights(ws):
        return png_density(ws, width, sigma)

    return joint_iterative_weighted_centroid(distal_data, proximal_data, geometry, weights, tol=tol, max_iter=max_iter)