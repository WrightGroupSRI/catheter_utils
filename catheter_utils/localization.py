"""Catheter localization algorithms. Take projection data and produce
coordinate estimates."""

import logging
from itertools import chain

import numpy
import scipy.ndimage
import scipy.optimize

import catheter_utils.geometry
import catheter_utils.projections

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
            logger.debug("converged in %s iterations", itr+1)
            break
    else:
        logger.debug("maximum iterations reached")

    return x0,itr+1


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

    @staticmethod
    def world_to_projection_mat():
        return numpy.eye(3)


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

    @classmethod
    def world_to_projection_mat(cls):
        return cls._BACKWARD


def localize_coil(data, localizer, localizer_args=None, localizer_kwargs=None):
    """A helper for combining single projection localizations into an estimate
    of a coil position.

    Returns the 3d coordinates and number of iterations required for each
    projection. For non-iterative algorithms, the number of iterations
    will be 0.
    """

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

    coords = [
        localizer(fs, xs, *localizer_args, **localizer_kwargs) for fs, xs in data
    ]
    iters = numpy.zeros(len(data))
    # Iterative functions return both coordinates and number of iterations
    if len(coords) > 0 and type(coords[0]) is tuple:
        iters = numpy.array([b for a,b in coords])
        coords = [a for a,b in coords]
    return coordinate_system.projection_to_world(numpy.array(coords)),iters


def localize_catheter(distal_data, proximal_data, localizer, localizer_args=None, localizer_kwargs=None):
    """A helper for combining single projection localizations into an estimate
    for two coil locations.

    Returns the distal coordinates, proximal coordinates, and the highest number
    of iterations required (zero for non-iterative algorithms)
    """

    distal,it_distal = localize_coil(distal_data, localizer, localizer_args, localizer_kwargs)
    proximal,it_prox = localize_coil(proximal_data, localizer, localizer_args, localizer_kwargs)
    # the largest number of iterations across coils & projections
    max_iter = numpy.max( numpy.concatenate((it_distal,it_prox)) )
    return distal, proximal, max_iter


class JointIterativeWeightedCentroid:
    """Responsible for localizing a catheter using the iterative centroid
    method with errors weighted according to a function of the projections.
    """

    def __init__(self, geometry=None, centroid_weighting=None, err_weighting=None, tol=1e-6, max_itr=10):
        """Args:
            geometry, the geometry of the catheter;
            centroid_weighting, the weighting used in the iterative centroid method (e.g., png);
            err_weighting, the weighting used to project the observed coordinates to get a fit (e.g., snr, variance);
            tol, the convergence criterion;
            max_itr, the maximum number of iterations to allow. """
        self.geometry = geometry or catheter_utils.geometry.GEOMETRY[-1]
        self.centroid_weighting = centroid_weighting or (lambda xs: png_density(xs))
        self.err_weighting = err_weighting
        self.tol = tol
        self.max_itr = max_itr

    def localize(self, distal_data, proximal_data):
        if len(distal_data) == 3 and len(proximal_data) == 3:
            coordinate_system = XYZCoordinates()
        elif len(distal_data) == 4 and len(proximal_data) == 4:
            coordinate_system = HadamardCoordinates()
        else:
            raise ValueError("Expected 3 or 4 projection directions")

        def target(data):
            coords = [
                iterative_weighted_centroid(fs, xs, self.centroid_weighting) for (fs, xs) in data
                ]
            if len(coords) > 0 and type(coords[0]) is tuple:
                coords = [a for a,b in coords]
            return numpy.array(coords)

        d0 = target(distal_data)
        p0 = target(proximal_data)

        if self.err_weighting:
            def calc_weights(data):
                ws = numpy.array([self.err_weighting(fs, xs) for fs, xs in data])
                ws = numpy.sqrt(ws) + 1e-8
                ws /= numpy.max(ws)
                ws = numpy.diag(1.0 / ws)
                return ws

            wd = calc_weights(distal_data)
            wp = calc_weights(proximal_data)
        else:
            wd = None
            wp = None

        for itr in range(self.max_itr):
            d1 = d0
            d = coordinate_system.world_to_projection(d1)
            d0 = coordinate_system.projection_to_world(
                numpy.array([centroid(fs * self.centroid_weighting(xs - x), xs) for (fs, xs), x in zip(distal_data, d)])
            )

            p1 = p0
            p = coordinate_system.world_to_projection(p1)
            p0 = coordinate_system.projection_to_world(
                numpy.array([centroid(fs * self.centroid_weighting(xs - x), xs) for (fs, xs), x in zip(proximal_data, p)])
            )

            if self.err_weighting:
                d0, p0 = ProjectionWeightedCatheterFit.fit(
                    coordinate_system.world_to_projection_mat(),
                    self.geometry.distal_to_proximal,
                    wd,
                    d0,
                    wp,
                    p0
                )

            else:
                fit = self.geometry.fit_from_coils_mse(d0, p0)
                d0 = fit.distal.reshape(-1)
                p0 = fit.proximal.reshape(-1)

            if numpy.linalg.norm(d0 - d1) <= self.tol and numpy.linalg.norm(p0 - p1) <= self.tol:
                logger.debug("converged in %s iterations", itr+1)
                break
        else:
            logger.debug("maximum iterations reached")
        #TODO: fix wjpng failing out on some inputs
        return d0, p0, itr


def joint_iterative_weighted_centroid(distal_data, proximal_data, geometry, weighting, tol=1e-6, max_iter=256):
    # This is the old joint iterative centroid fn that doesn't use any error weighting.
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
            logger.debug("converged in %s iterations", itr+1)
            break
    else:
        logger.debug("maximum iterations reached")

    return d0, p0, itr+1


def jpng(distal_data, proximal_data, geometry, width=3, sigma=0.5, tol=1e-6, max_iter=256):

    def weights(ws):
        return png_density(ws, width, sigma)

    return joint_iterative_weighted_centroid(distal_data, proximal_data, geometry, weights, tol=tol, max_iter=max_iter)


class ProjectionWeightedCatheterFit:
    """This object is responsible for fitting the distal and proximal coils to
    a sample of observed coordinates while weighting errors according to the
    given weights."""

    def __init__(self, world_to_projection, intercoil_distance, projection_weights_a, a, projection_weights_b, b):
        """Args:
            world_to_projection, the projection matrix that sends coil coordinates to projection coordinates;
            intercoil_distance, the distance between the coils;
            projection_weights_a, the weights used to measure fit error for coil a;
            projection_weights_b, the weights used to measure fit error for coil b;
            a, the observed coordinate of coil a;
            b, the observed coordinate of coil b;
        """

        pa = projection_weights_a @ world_to_projection
        pb = projection_weights_b @ world_to_projection

        self.intercoil_distance = intercoil_distance
        self.na = pa.T @ pa  # n for normal
        self.nb = pb.T @ pb  #
        self.ba = numpy.linalg.inv(pa.T @ pa)
        self.bb = numpy.linalg.inv(pb.T @ pb)
        self.target_a = pa.T @ (projection_weights_a @ a)
        self.target_b = pb.T @ (projection_weights_b @ b)

    @staticmethod
    def _unpack_x(x):
        return x[0:3], x[3:6], x[6]

    def fun(self, x):
        """The lagrangian function for the weighted minimum norm problem (mse
        estimate) with the inter-coil distance constraint."""
        # There should be a PDF going into more detail.
        a, b, h = self._unpack_x(x)
        delta = a - b
        return numpy.concatenate([
            self.na @ a - self.target_a - h * delta,
            self.nb @ b - self.target_b + h * delta,
            [0.5 * (numpy.dot(delta, delta) - self.intercoil_distance ** 2)],
        ])

    @property
    def init(self):
        """The initial guess for the location of the root. Use the unconstrained
        estimate."""
        return numpy.concatenate([
            self.ba @ self.target_a,  # unconstrained estimate,
            self.bb @ self.target_b,
            [0],
        ])

    def solve(self):
        """Find a root of the lagrangian."""
        res = scipy.optimize.root(
            self.fun,
            self.init
        )
        a, b, h = self._unpack_x(res["x"])
        return a, b

    @classmethod
    def fit(cls, p, c, wa, a, wb, b):
        """Set up the root finding problem and find the root."""
        l = cls(p, c, wa, a, wb, b)
        res = scipy.optimize.root(
            l.fun,
            l.init,
        )
        a1 = res["x"][0:3]
        b1 = res["x"][3:6]
        # h1 = res["x"][6]
        return a1, b1
