"""This module contains things related to the geometry of our MR compatible
catheters. Calculations involving specific information about their measurements
can be put in here. E.g., extrapolating the tip from observations of the
tracking coil locations."""

import numpy
from collections import namedtuple

# Catheter landmarks:
#   <-----*---x---*--
#   t     d   m   p
#
# t: tip location
# d: distal coil location
# p: proximal coil location
# m: midpoint between catheter coils


class Geometry(namedtuple("_GeometryBase", "tip_to_distal tip_to_proximal")):
    """The measurements that describe the geometry of a catheter."""

    @property
    def distal_to_proximal(self):
        return self.tip_to_proximal - self.tip_to_distal

    @property
    def distal_to_midpoint(self):
        return 0.5 * self.distal_to_proximal

    @property
    def tip_to_midpoint(self):
        return self.tip_to_distal + self.distal_to_midpoint

    FitFromCoilsResult = namedtuple("FitFromCoilsResult", "tip distal proximal")
    """The various methods of fitting catheter coordinate locations from
    coil observations all return the same kind of data."""

    def fit_from_coils_mse(self, distal, proximal):
        """Extrapolate the tip location from the distal and proximal coil
        locations by finding the minimum error coil locations and then
        extrapolating the tip."""
        distal = distal.reshape((-1, 3))
        proximal = proximal.reshape((-1, 3))

        direction = distal - proximal
        norms = numpy.linalg.norm(direction, axis=1).reshape((direction.shape[0], 1))
        norms += 1e-8

        direction /= norms
        midpoint = 0.5 * (distal + proximal)

        tip = midpoint + self.tip_to_midpoint * direction
        distal = midpoint + self.distal_to_midpoint * direction
        proximal = midpoint - self.distal_to_midpoint * direction

        return self.FitFromCoilsResult(tip=tip, distal=distal, proximal=proximal)

    def fit_from_coils_distal_offset(self, distal, proximal):
        """Extrapolate the tip location from the distal and proximal coil
        locations by adding an offset to the distal coil location. This
        is the method used historically by our realtime pipeline."""
        distal = distal.reshape((-1, 3))
        proximal = proximal.reshape((-1, 3))

        direction = distal - proximal
        norms = numpy.linalg.norm(direction, axis=1).reshape((direction.shape[0], 1))
        norms += 1e-8

        direction /= norms

        distal = distal.copy()
        tip = distal + self.tip_to_distal * direction
        proximal = distal - self.distal_to_proximal * direction

        return self.FitFromCoilsResult(tip=tip, distal=distal, proximal=proximal)


def estimate_geometry(distal, proximal):
    """Given a set of catheter coordinates, find the catheter specifications
    that best fit the data."""
    # There is approx a 0.79 mm difference between the existing catheters,
    # dunno if that is too small to categorize geometry this way.
    # TODO
    #   Someone should do a quick check. Could add a warning if there aren't
    #   enough points to reliably make a determination.
    mean_distal_to_proximal = numpy.mean(numpy.linalg.norm(proximal-distal, axis=1))
    return min(GEOMETRY, key=lambda g: abs(g.distal_to_proximal - mean_distal_to_proximal))


GEOMETRY = tuple([
    Geometry(tip_to_distal=(10.69 + 8.13) / 2, tip_to_proximal=(19.25 + 16.69) / 2),  # spec acquired in 2018
    Geometry(tip_to_distal=(10.07 + 8.07) / 2, tip_to_proximal=(17.84 + 15.84) / 2),  # spec acquired in 2019
])
"""Measurements of each of our models of catheter in the order that we received them."""
