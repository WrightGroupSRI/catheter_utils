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


def _cc(us, vs):
    """Calculate the correlation coefficient between the given signals."""
    n = numpy.trapz(us * vs)
    d = numpy.sqrt(numpy.trapz(us * us) * numpy.trapz(vs * vs))
    return n / d if d > 0.0 else 0.0


def _cc_objective(fs, xs, template):
    """Produce a function that maps an 'x' coordinate to the correlation of
    the given signal with the template centered at that coordinate."""

    def _obj(x):
        return _cc(fs, template(xs - x))

    return _obj


class XYZCoordinates:
    """Convert between XYZ projection ("sri") projection and scanner
    coordinates."""

    @staticmethod
    def projection_to_scanner(x):
        """Transform from `projection` coordinates to `scanner`
        coordinates."""
        return x

    @staticmethod
    def scanner_to_projection(u):
        """Transform from `scanner` coordinates to `projection`
        coordinates."""
        return u


class HadamardCoordinates:
    """Convert between Hadamard projection coordinates ("fh") and scanner
    coordinates."""

    # s2p, 3 scanner coordinates -> 4 projection coordinates
    # Each 'projection' coordinate is calculated by projecting
    # the scanner coordinate onto the corresponding direction
    # unit vector.
    _BACKWARD = (3**-0.5) * numpy.array([
        [-1, -1, -1],
        [1, 1, -1],
        [1, -1, 1],
        [-1, 1, 1]
    ])

    # p2s, 4 projection coordinates -> 3 scanner coordinates
    # Least squares solution provided by pinv.
    _FORWARD = numpy.linalg.pinv(_BACKWARD)

    @classmethod
    def projection_to_scanner(cls, x):
        """Transform from `projection` coordinates to `scanner`
        coordinates."""
        return cls._FORWARD @ x

    @classmethod
    def scanner_to_projection(cls, u):
        """Transform from `scanner` coordinates to `projection`
        coordinates."""
        return cls._BACKWARD @ u


def correlate_template(data, coordinate_system, template):
    # Data should be a list of tuples (fs, xs)

    # Initial guess close to peak locations
    init = coordinate_system.projection_to_scanner(numpy.array([
        peak(fs, xs) for fs, xs in data
    ]))

    obj_components = [
        _cc_objective(fs, xs, template) for fs, xs in data
    ]

    def obj(x):
        u = coordinate_system.scanner_to_projection(x)
        return sum(f(u[axis]) for axis, f in enumerate(obj_components))

    res = scipy.optimize.minimize(
        obj,
        init,
        method="SLSQP",
    )

    return res['x']


def joint_correlate_template(distal_data, proximal_data, coordinate_system, geometry, template):

    # initialize near the peak location
    init_distal = coordinate_system.projection_to_scanner(numpy.array([
        peak(fs, xs) for fs, xs in distal_data
    ]))
    init_proximal = coordinate_system.projection_to_scanner(numpy.array([
        peak(fs, xs) for fs, xs in proximal_data
    ]))

    # Make sure that initial guess satisfies the constraint
    init_fit = geometry.fit_from_coils_mse(init_distal, init_proximal)
    init = numpy.concatenate((init_fit.distal, init_fit.proximal), axis=1).reshape(-1)

    obj_distal_components = [
        _cc_objective(scipy.ndimage.gaussian_filter1d(fs, sigma=1), xs, template) for fs, xs in distal_data
    ]
    obj_proximal_components = [
        _cc_objective(scipy.ndimage.gaussian_filter1d(fs, sigma=1), xs, template) for fs, xs in proximal_data
    ]

    def obj(x):
        u1 = coordinate_system.scanner_to_projection(x[0:3])
        s1 = sum(f(u1[axis]) for axis, f in enumerate(obj_distal_components))
        u2 = coordinate_system.scanner_to_projection(x[3:6])
        s2 = sum(f(u2[axis]) for axis, f in enumerate(obj_proximal_components))
        return s1 + s2

    def constraint(x):
        return numpy.linalg.norm(x[0:3] - x[3:6]) - geometry.distal_to_proximal

    res = scipy.optimize.minimize(
        obj,
        init,
        method="trust-constr",
        constraints={"type": "eq", "fun": constraint},
    )

    res_x = res["x"]
    return res_x[0:3], res_x[3:6]


def cpng(data):
    """Localize the catheter by correlating the PNG density with the signal."""

    if len(data) == 3:
        coordinate_system = XYZCoordinates()
    elif len(data) == 4:
        coordinate_system = HadamardCoordinates()
    else:
        raise ValueError("Expected 3 or 4 projection directions")

    return correlate_template(
        data,
        coordinate_system,
        lambda xs: png_density(xs)
    )


def jpng(distal_data, proximal_data, geometry):
    """Localize the catheter by correlating the PNG density with the signal
    while applying the known catheter geometry's constraints."""

    if len(distal_data) == 3 and len(proximal_data) == 3:
        coordinate_system = XYZCoordinates()
    elif len(distal_data) == 4 and len(proximal_data) == 4:
        coordinate_system = HadamardCoordinates()
    else:
        raise ValueError("Expected 3 or 4 projection directions")

    return joint_correlate_template(
        distal_data,
        proximal_data,
        coordinate_system,
        geometry,
        lambda xs: png_density(xs)
    )


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

    return coordinate_system.projection_to_scanner(numpy.array([
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

    distal0 = coordinate_system.projection_to_scanner(numpy.array([
        png(fs, xs) for fs, xs in distal_data
    ]))
    proximal0 = coordinate_system.projection_to_scanner(numpy.array([
        png(fs, xs) for fs, xs in proximal_data
    ]))

    for itr in range(max_iter):
        distal1 = distal0
        distal = coordinate_system.scanner_to_projection(distal1)
        distal0 = coordinate_system.projection_to_scanner(
            numpy.array([centroid(fs * weighting(xs - x), xs) for (fs, xs), x in zip(distal_data, distal)])
        )

        proximal1 = proximal0
        proximal = coordinate_system.scanner_to_projection(proximal1)
        proximal0 = coordinate_system.projection_to_scanner(
            numpy.array([centroid(fs * weighting(xs - x), xs) for (fs, xs), x in zip(proximal_data, proximal)])
        )

        fit = geometry.fit_from_coils_mse(distal0, proximal0)
        distal0 = fit.distal.reshape(-1)
        proximal0 = fit.proximal.reshape(-1)

        if numpy.linalg.norm(distal0 - distal1) + numpy.linalg.norm(proximal0 - proximal1) <= tol:
            logger.debug("converged in %s iterations", itr)
            break
    else:
        logger.debug("maximum iterations reached")

    return distal0, proximal0


def jpo(distal_data, proximal_data, geometry, width=3, sigma=0.5, tol=1e-6, max_iter=256):

    def weights(ws):
        return png_density(ws, width, sigma)

    return joint_iterative_weighted_centroid(distal_data, proximal_data, geometry, weights, tol=tol, max_iter=max_iter)
