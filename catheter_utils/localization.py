"""Catheter localization algorithms. Take projection data and produce
coordinate estimates."""

import logging
import numpy
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


def png_density(xs, width, sigma):
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


# def _cc(us, vs):
#     """Calculate the correlation coefficient between the given signals."""
#     n = numpy.trapz(us * vs)
#     d = numpy.sqrt(numpy.trapz(us * us) * numpy.trapz(vs * vs))
#     return n / d if d > 0.0 else 0.0
#
#
# def _cc_objective(fs, xs, template):
#     """Produce a function that maps an 'x' coordinate to the correlation of
#     the given signal with the template centered at that coordinate."""
#
#     def _obj(x):
#         return _cc(fs, template(xs - x))
#
#     return _obj
#
#
# def correlate_template(fs, xs, template):
#     obj = _cc_objective(fs, xs, template)
#     res = scipy.optimize.minimize(obj, peak(fs, xs), method="SLSQP", bounds=(xs[0], xs[-1]))
#     return res['x']

