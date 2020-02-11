"""Test catheter_utils.cathcoords."""

import numpy
import numpy.testing
from catheter_utils import geometry
from textwrap import dedent
from unittest import TestCase
from unittest.mock import patch, mock_open

_DISTAL = numpy.array([
    [0.0, 0, 0],
    [0.0, 0, 0],
])

_PROXIMAL = numpy.array([
    [10.0, 0, 0],
    [10.0, 0, 0],
])

_GEOMETRY = geometry.Geometry(tip_to_distal=1.0, tip_to_proximal=13.0)
_GEOMETRY2 = geometry.Geometry(tip_to_distal=1.0, tip_to_proximal=10.0)


class GeometryTests(TestCase):

    def setUp(self):
        self.geometry = _GEOMETRY

    def test_fit_mse(self):
        t, d, p = self.geometry.fit_from_coils_mse(_DISTAL, _PROXIMAL)
        numpy.testing.assert_array_almost_equal(t, numpy.array([[-2.0, 0, 0], [-2.0, 0, 0]]))
        numpy.testing.assert_array_almost_equal(d, numpy.array([[-1.0, 0, 0], [-1.0, 0, 0]]))
        numpy.testing.assert_array_almost_equal(p, numpy.array([[11.0, 0, 0], [11.0, 0, 0]]))

    def test_fit_distal_offset(self):
        t, d, p = self.geometry.fit_from_coils_distal_offset(_DISTAL, _PROXIMAL)
        numpy.testing.assert_array_almost_equal(t, numpy.array([[-1.0, 0, 0], [-1.0, 0, 0]]))
        numpy.testing.assert_array_almost_equal(d, numpy.array([[-0.0, 0, 0], [-0.0, 0, 0]]))
        numpy.testing.assert_array_almost_equal(p, numpy.array([[12.0, 0, 0], [12.0, 0, 0]]))

    @patch("catheter_utils.geometry.GEOMETRY",
           tuple([_GEOMETRY, _GEOMETRY2, *geometry.GEOMETRY]))
    def test_estimate_geometry(self):
        g = geometry.estimate_geometry(_DISTAL, _PROXIMAL)
        self.assertEqual(g, _GEOMETRY2)
